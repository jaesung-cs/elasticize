#include <elasticize/gpu/engine.h>

#include <iostream>
#include <fstream>

#include <elasticize/window/window_manager.h>
#include <elasticize/window/window.h>
#include <elasticize/gpu/buffer.h>

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
  destroySwapchain();
  destroyDevice();
  destroyInstance();
}

vk::Buffer Engine::createBuffer(vk::DeviceSize size)
{
  const auto bufferInfo = vk::BufferCreateInfo()
    .setUsage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst)
    .setSize(size);

  auto buffer = device_.createBuffer(bufferInfo);

  const auto memoryRequirements = device_.getBufferMemoryRequirements(buffer);
  deviceMemoryOffset_ = align(deviceMemoryOffset_, memoryRequirements.alignment);
  device_.bindBufferMemory(buffer, deviceMemory_, deviceMemoryOffset_);
  deviceMemoryOffset_ += size;

  return buffer;
}

void Engine::destroyBuffer(vk::Buffer buffer)
{
  device_.destroyBuffer(buffer);
}

void Engine::transferToGpu(const void* data, vk::DeviceSize size, vk::Buffer buffer)
{
  // To staging buffer
  std::memcpy(stagingBufferMap_, data, size);

  // To target buffer
  const auto region = vk::BufferCopy()
    .setSrcOffset(0)
    .setDstOffset(0)
    .setSize(size);

  const auto allocateInfo = vk::CommandBufferAllocateInfo()
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandPool(transientCommandPool_)
    .setCommandBufferCount(1);

  const auto cb = device_.allocateCommandBuffers(allocateInfo)[0];

  cb.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
  cb.copyBuffer(stagingBuffer_, buffer, region);
  cb.end();

  const auto submit = vk::SubmitInfo().setCommandBuffers(cb);
  queue_.submit(submit, transferFence_);

  // TODO: don't wait for transfer completion
  device_.waitForFences(transferFence_, true, UINT64_MAX);
  device_.resetFences(transferFence_);
  device_.freeCommandBuffers(transientCommandPool_, cb);
}

void Engine::transferFromGpu(void* data, vk::DeviceSize size, vk::Buffer buffer)
{
  // To target buffer
  const auto region = vk::BufferCopy()
    .setSrcOffset(0)
    .setDstOffset(0)
    .setSize(size);

  const auto allocateInfo = vk::CommandBufferAllocateInfo()
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandPool(transientCommandPool_)
    .setCommandBufferCount(1);

  const auto cb = device_.allocateCommandBuffers(allocateInfo)[0];

  cb.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
  cb.copyBuffer(buffer, stagingBuffer_, region);
  cb.end();

  const auto submit = vk::SubmitInfo().setCommandBuffers(cb);
  queue_.submit(submit, transferFence_);

  // TODO: don't wait for transfer completion
  device_.waitForFences(transferFence_, true, UINT64_MAX);
  device_.resetFences(transferFence_);
  device_.freeCommandBuffers(transientCommandPool_, cb);

  // To staging buffer
  std::memcpy(data, stagingBufferMap_, size);
}

void Engine::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize byteSize)
{
  // To target buffer
  const auto region = vk::BufferCopy()
    .setSrcOffset(0)
    .setDstOffset(0)
    .setSize(byteSize);

  const auto allocateInfo = vk::CommandBufferAllocateInfo()
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandPool(transientCommandPool_)
    .setCommandBufferCount(1);

  const auto cb = device_.allocateCommandBuffers(allocateInfo)[0];

  cb.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
  cb.copyBuffer(srcBuffer, dstBuffer, region);
  cb.end();

  const auto submit = vk::SubmitInfo().setCommandBuffers(cb);
  queue_.submit(submit, transferFence_);

  // TODO: don't wait for transfer completion
  device_.waitForFences(transferFence_, true, UINT64_MAX);
  device_.resetFences(transferFence_);
  device_.freeCommandBuffers(transientCommandPool_, cb);
}

void Engine::attachWindow(const window::Window& window)
{
  destroySwapchain();

  surface_ = window.createVulkanSurface(instance_);

  createSwapchain(window);
}

void Engine::addComputeShader(const std::string& filepath)
{
  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings(3);
  bindings[0]
    .setBinding(0)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[1]
    .setBinding(1)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  bindings[2]
    .setBinding(2)
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1);

  const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo().setBindings(bindings);
  const auto descriptorSetLayout = device_.createDescriptorSetLayout(descriptorSetLayoutInfo);

  // Pipeline layout
  std::vector<vk::PushConstantRange> pushConstantRange(1);
  pushConstantRange[0]
    .setStageFlags(vk::ShaderStageFlagBits::eCompute)
    .setOffset(0)
    .setSize(sizeof(uint32_t) * 3);

  const auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
    .setSetLayouts(descriptorSetLayout)
    .setPushConstantRanges(pushConstantRange);

  const auto pipelineLayout = device_.createPipelineLayout(pipelineLayoutInfo);

  // Pipeline
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

  const auto stage = vk::PipelineShaderStageCreateInfo()
    .setStage(vk::ShaderStageFlagBits::eCompute)
    .setModule(module)
    .setPName("main");

  const auto pipelineInfo = vk::ComputePipelineCreateInfo()
    .setLayout(pipelineLayout)
    .setStage(stage);

  const auto pipeline = device_.createComputePipeline(nullptr, pipelineInfo).value;

  device_.destroyShaderModule(module);

  // Add pipeline
  ComputePipeline computePipeline;
  computePipeline.descriptorSetLayout = descriptorSetLayout;
  computePipeline.pipelineLayout = pipelineLayout;
  computePipeline.pipeline = pipeline;
  computePipelines_.push_back(computePipeline);
}

void Engine::addDescriptorSet(vk::Buffer arrayBuffer, vk::Buffer counterBuffer, vk::Buffer outBuffer)
{
  // TODO: receive descriptor set layout and buffers from parameters
  const auto descriptorSetLayout = computePipelines_[0].descriptorSetLayout;
  std::vector<vk::Buffer> buffers = {
    arrayBuffer,
    counterBuffer,
    outBuffer,
  };

  const auto descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo()
    .setDescriptorPool(descriptorPool_)
    .setSetLayouts(descriptorSetLayout);
  const auto descriptorSet = device_.allocateDescriptorSets(descriptorSetAllocateInfo)[0];

  std::vector<vk::DescriptorBufferInfo> bufferInfos(3);
  bufferInfos[0]
    .setBuffer(buffers[0])
    .setOffset(0)
    .setRange(VK_WHOLE_SIZE);

  bufferInfos[1]
    .setBuffer(buffers[1])
    .setOffset(0)
    .setRange(VK_WHOLE_SIZE);

  bufferInfos[2]
    .setBuffer(buffers[2])
    .setOffset(0)
    .setRange(VK_WHOLE_SIZE);

  std::vector<vk::WriteDescriptorSet> writes(3);
  writes[0]
    .setDstBinding(0)
    .setDstSet(descriptorSet)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1)
    .setBufferInfo(bufferInfos[0]);

  writes[1]
    .setDstBinding(1)
    .setDstSet(descriptorSet)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1)
    .setBufferInfo(bufferInfos[1]);

  writes[2]
    .setDstBinding(2)
    .setDstSet(descriptorSet)
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
    .setDescriptorCount(1)
    .setBufferInfo(bufferInfos[2]);

  device_.updateDescriptorSets(writes, {});

  descriptorSets_.push_back(descriptorSet);
}

void Engine::runComputeShader(int computeShaderId, int n, int bitOffset, int scanOffset)
{
  constexpr auto BLOCK_SIZE = 256;
  constexpr auto RADIX_SIZE = 256;
  const auto groupSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const auto allocateInfo = vk::CommandBufferAllocateInfo()
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandPool(transientCommandPool_)
    .setCommandBufferCount(1);

  const auto cb = device_.allocateCommandBuffers(allocateInfo)[0];

  cb.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
  cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelines_[computeShaderId].pipelineLayout, 0, { descriptorSets_[0] }, {});
  cb.bindPipeline(vk::PipelineBindPoint::eCompute, computePipelines_[computeShaderId].pipeline);
  cb.pushConstants<int>(computePipelines_[computeShaderId].pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, n);
  cb.pushConstants<int>(computePipelines_[computeShaderId].pipelineLayout, vk::ShaderStageFlagBits::eCompute, sizeof(n), bitOffset);
  cb.pushConstants<int>(computePipelines_[computeShaderId].pipelineLayout, vk::ShaderStageFlagBits::eCompute, sizeof(n) + sizeof(bitOffset), scanOffset);
  cb.dispatch(groupSize, 1, 1);
  cb.end();

  const auto submit = vk::SubmitInfo().setCommandBuffers(cb);
  queue_.submit(submit, transferFence_);

  // TODO: don't wait for compute job completion
  device_.waitForFences(transferFence_, true, UINT64_MAX);
  device_.resetFences(transferFence_);
  device_.freeCommandBuffers(transientCommandPool_, cb);
}

void Engine::createSwapchain(const window::Window& window)
{
  // Swapchain
  if (!physicalDevice_.getSurfaceSupportKHR(queueIndex_, surface_))
    throw std::runtime_error("Device queue does not support surface");

  const auto capabilities = physicalDevice_.getSurfaceCapabilitiesKHR(surface_);

  // Triple buffering
  auto imageCount = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    imageCount = capabilities.maxImageCount;
  if (imageCount != 3)
    throw std::runtime_error("Triple buffering is not supported");

  // Present mode: use mailbox if available. Limit fps in draw call
  vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
  const auto presentModes = physicalDevice_.getSurfacePresentModesKHR(surface_);
  for (auto availableMode : presentModes)
  {
    if (availableMode == vk::PresentModeKHR::eMailbox)
    {
      presentMode = vk::PresentModeKHR::eMailbox;
      break;
    }
  }

  // Swapchain format
  const auto availableFormats = physicalDevice_.getSurfaceFormatsKHR(surface_);
  auto format = availableFormats[0];
  for (const auto& availableFormat : availableFormats)
  {
    if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
      availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
      format = availableFormat;
  }

  // Swapchain extent
  vk::Extent2D extent;
  if (capabilities.currentExtent.width != UINT32_MAX)
    extent = capabilities.currentExtent;
  else
  {
    const auto width = window.width();
    const auto height = window.height();

    VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
    extent = actualExtent;
  }

  constexpr auto usage = vk::ImageUsageFlagBits::eColorAttachment;

  swapchainInfo_ = vk::SwapchainCreateInfoKHR()
    .setSurface(surface_)
    .setMinImageCount(imageCount)
    .setImageFormat(format.format)
    .setImageColorSpace(format.colorSpace)
    .setImageExtent(extent)
    .setImageArrayLayers(1)
    .setImageUsage(usage)
    .setImageSharingMode(vk::SharingMode::eExclusive)
    .setPreTransform(capabilities.currentTransform)
    .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
    .setPresentMode(presentMode)
    .setClipped(true);

  swapchain_ = device_.createSwapchainKHR(swapchainInfo_);
}

void Engine::destroySwapchain()
{
  if (swapchain_)
    device_.destroySwapchainKHR(swapchain_);

  if (surface_)
    instance_.destroySurfaceKHR(surface_);
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
    .setFlags(vk::CommandPoolCreateFlagBits::eTransient);

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

  const auto deviceInfo = vk::DeviceCreateInfo()
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
