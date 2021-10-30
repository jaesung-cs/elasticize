#include <elasticize/gpu/engine.h>

#include <iostream>

#include <elasticize/window/window_manager.h>
#include <elasticize/window/window.h>

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
}

Engine::Engine(Options options)
  : options_(options)
{
  createInstance();
  createDevice();
}

Engine::~Engine()
{
  device_.waitIdle();

  destroySwapchain();
  destroyDevice();
  destroyInstance();
}

void Engine::attachWindow(const window::Window& window)
{
  destroySwapchain();

  surface_ = window.createVulkanSurface(instance_);

  createSwapchain(window);
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
  {
    instance_ = vk::createInstance(instanceInfo);
  }
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
