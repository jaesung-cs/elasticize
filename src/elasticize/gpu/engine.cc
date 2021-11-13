#include <elasticize/gpu/engine.h>

#include <iostream>
#include <fstream>

#include <elasticize/window/window_manager.h>
#include <elasticize/window/window.h>
#include <elasticize/gpu/buffer.h>
#include <elasticize/gpu/image.h>

namespace elastic
{
namespace gpu
{
namespace
{
// Validation layer callback
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  VkDebugUtilsMessageTypeFlagsEXT messageType,
  const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
  void* pUserData)
{
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    std::cerr << callbackData->pMessage << std::endl << std::endl;

  return VK_FALSE;
}

vk::DeviceSize align(vk::DeviceSize offset, vk::DeviceSize alignment)
{
  return (offset + alignment - 1) & ~(alignment - 1);
}
}

Engine::Engine(Options options)
  : options_(options)
{
  createInstance();
  createDevice();
  createMemoryPool();
  createCommandPool();
  createDescriptorPool();
}

Engine::~Engine()
{
  device_.waitIdle();

  for (auto& computePipeline : computePipelines_)
  {
    device_.destroyDescriptorSetLayout(computePipeline.descriptorSetLayout);
    device_.destroyPipelineLayout(computePipeline.pipelineLayout);
    device_.destroyPipeline(computePipeline.pipeline);
  }

  destroyDescriptorPool();
  destroyCommandPool();
  destroyMemoryPool();
  destroyDevice();
  destroyInstance();
}

vk::Buffer Engine::createBuffer(vk::DeviceSize size)
{
  const auto bufferInfo = vk::BufferCreateInfo()
    .setUsage(
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
      vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer)
    .setSize(size);

  auto buffer = device_.createBuffer(bufferInfo);

  const auto memoryRequirements = device_.getBufferMemoryRequirements(buffer);
  deviceMemoryOffset_ = align(deviceMemoryOffset_, memoryRequirements.alignment);
  device_.bindBufferMemory(buffer, deviceMemory_, deviceMemoryOffset_);
  deviceMemoryOffset_ += size;

  return buffer;
}

void Engine::bindImageMemory(vk::Image image)
{
  const auto memoryRequirements = device_.getImageMemoryRequirements(image);
  deviceMemoryOffset_ = align(deviceMemoryOffset_, memoryRequirements.alignment);
  device_.bindImageMemory(image, deviceMemory_, deviceMemoryOffset_);
  deviceMemoryOffset_ += memoryRequirements.size;
}

vk::ShaderModule Engine::createShaderModule(const std::string& filepath)
{
  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filepath);

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  std::vector<uint32_t> code;
  auto* intPtr = reinterpret_cast<uint32_t*>(buffer.data());
  for (int i = 0; i < fileSize / 4; i++)
    code.push_back(intPtr[i]);

  const auto shaderModuleInfo = vk::ShaderModuleCreateInfo().setCode(code);
  const auto module = device_.createShaderModule(shaderModuleInfo);

  return module;
}

void Engine::createMemoryPool()
{
  const auto memoryProperties = physicalDevice_.getMemoryProperties();

  // Find memroy type index
  uint64_t deviceAvailableSize = 0;
  uint64_t hostAvailableSize = 0;
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
  {
    const auto properties = memoryProperties.memoryTypes[i].propertyFlags;
    const auto heapIndex = memoryProperties.memoryTypes[i].heapIndex;
    const auto heap = memoryProperties.memoryHeaps[heapIndex];

    if ((properties & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)
    {
      if (heap.size > deviceAvailableSize)
      {
        deviceIndex_ = i;
        deviceAvailableSize = heap.size;
      }
    }

    if ((properties & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))
      == (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))
    {
      if (heap.size > hostAvailableSize)
      {
        hostIndex_ = i;
        hostAvailableSize = heap.size;
      }
    }
  }

  {
    const auto allocateInfo = vk::MemoryAllocateInfo()
      .setMemoryTypeIndex(deviceIndex_)
      .setAllocationSize(options_.memoryPoolSize);
    deviceMemory_ = device_.allocateMemory(allocateInfo);
  }
  {
    const auto allocateInfo = vk::MemoryAllocateInfo()
      .setMemoryTypeIndex(hostIndex_)
      .setAllocationSize(options_.memoryPoolSize);
    hostMemory_ = device_.allocateMemory(allocateInfo);

    const auto bufferInfo = vk::BufferCreateInfo()
      .setUsage(vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst)
      .setSize(allocateInfo.allocationSize);
    stagingBuffer_ = device_.createBuffer(bufferInfo);

    device_.bindBufferMemory(stagingBuffer_, hostMemory_, 0);

    stagingBufferMap_ = reinterpret_cast<uint8_t*>(device_.mapMemory(hostMemory_, 0, allocateInfo.allocationSize));
  }
}

void Engine::destroyMemoryPool()
{
  device_.unmapMemory(hostMemory_);
  stagingBufferMap_ = nullptr;
  device_.destroyBuffer(stagingBuffer_);
  device_.freeMemory(deviceMemory_);
  device_.freeMemory(hostMemory_);
}

void Engine::createCommandPool()
{
  const auto commandPoolInfo = vk::CommandPoolCreateInfo()
    .setQueueFamilyIndex(queueIndex_)
    .setFlags(vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

  transientCommandPool_ = device_.createCommandPool(commandPoolInfo);
  transferFence_ = device_.createFence({});
}

void Engine::destroyCommandPool()
{
  device_.destroyFence(transferFence_);
  device_.destroyCommandPool(transientCommandPool_);
}

void Engine::createDescriptorPool()
{
  constexpr uint32_t maxSets = 256;
  constexpr uint32_t maxTypeCount = 256;

  std::vector<vk::DescriptorPoolSize> poolSizes = {
    {vk::DescriptorType::eStorageBuffer, maxTypeCount},
  };

  const auto descriptorPoolInfo = vk::DescriptorPoolCreateInfo()
    .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
    .setPoolSizes(poolSizes)
    .setMaxSets(maxSets);

  descriptorPool_ = device_.createDescriptorPool(descriptorPoolInfo);
}

void Engine::destroyDescriptorPool()
{
  device_.destroyDescriptorPool(descriptorPool_);
}

void Engine::createInstance()
{
  std::vector<const char*> layers;
  std::vector<const char*> instanceExtensions;
  if (options_.validationLayer)
  {
    layers.push_back("VK_LAYER_KHRONOS_validation");
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  if (!options_.headless)
  {
    const auto windowInstanceExtensions = window::WindowManager::requiredInstanceExtensions();
    for (const char* windowInstanceExtension : windowInstanceExtensions)
      instanceExtensions.push_back(windowInstanceExtension);
  }

  const auto appInfo = vk::ApplicationInfo()
    .setPApplicationName("Elasticize")
    .setApplicationVersion(1)
    .setPEngineName("Elasticize Engine")
    .setEngineVersion(1)
    .setApiVersion(VK_API_VERSION_1_2);

  const auto instanceInfo = vk::InstanceCreateInfo()
    .setFlags(vk::InstanceCreateFlags())
    .setPApplicationInfo(&appInfo)
    .setPEnabledExtensionNames(instanceExtensions)
    .setPEnabledLayerNames(layers);

  if (options_.validationLayer)
  {
    const auto messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT()
      .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose)
      .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
      .setPfnUserCallback(debugCallback);

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain{
      instanceInfo, messengerInfo
    };
    instance_ = vk::createInstance(chain.get<vk::InstanceCreateInfo>());

    // Create messneger
    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    vk::DispatchLoaderDynamic dld{ instance_, vkGetInstanceProcAddr };
    messenger_ = instance_.createDebugUtilsMessengerEXT(chain.get<vk::DebugUtilsMessengerCreateInfoEXT>(), nullptr, dld);
  }
  else
    instance_ = vk::createInstance(instanceInfo);
}

void Engine::destroyInstance()
{
  if (messenger_)
  {
    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    vk::DispatchLoaderDynamic dld{ instance_, vkGetInstanceProcAddr };
    instance_.destroyDebugUtilsMessengerEXT(messenger_, nullptr, dld);
  }

  instance_.destroy();
}

void Engine::selectSuitablePhysicalDevice()
{
  // TODO: choose among candidates, now assuming one GPU available
  physicalDevice_ = instance_.enumeratePhysicalDevices()[0];

  // Print physical device info
  const auto subgroupProperties = physicalDevice_.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>()
    .get<vk::PhysicalDeviceSubgroupProperties>();

  std::cout << "Subgroup properties:" << std::endl
    << "  subgroup size                : " << subgroupProperties.subgroupSize << std::endl
    << "  supported stages             : " << vk::to_string(subgroupProperties.supportedStages) << std::endl
    << "  supported operations         : " << vk::to_string(subgroupProperties.supportedOperations) << std::endl
    << "  quad operations in all stages: " << subgroupProperties.quadOperationsInAllStages << std::endl
    << std::endl;
}

void Engine::createDevice()
{
  selectSuitablePhysicalDevice();

  std::vector<const char*> deviceExtensions;

  if (!options_.headless)
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  const auto queueFamilyProperties = physicalDevice_.getQueueFamilyProperties();
  queueIndex_ = 0;
  for (int i = 0; i < queueFamilyProperties.size(); i++)
  {
    const auto& queueFamilyProperty = queueFamilyProperties[i];
    constexpr auto requiredQueueFlags = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
    if ((queueFamilyProperty.queueFlags & requiredQueueFlags) == requiredQueueFlags)
    {
      queueIndex_ = i;
      break;
    }
  }

  float queuePriorities[1] = {
    1.f,
  };
  std::vector<vk::DeviceQueueCreateInfo> queueInfos(1);
  queueInfos[0] = vk::DeviceQueueCreateInfo()
    .setQueueCount(1)
    .setQueueFamilyIndex(queueIndex_)
    .setPQueuePriorities(queuePriorities);

  auto deviceInfo = vk::DeviceCreateInfo()
    .setPEnabledExtensionNames(deviceExtensions)
    .setQueueCreateInfos(queueInfos);

  device_ = physicalDevice_.createDevice(deviceInfo);
  queue_ = device_.getQueue(queueIndex_, 0);
}

void Engine::destroyDevice()
{
  device_.destroy();
}
}
}
