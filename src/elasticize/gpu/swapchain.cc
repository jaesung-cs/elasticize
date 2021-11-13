#include <elasticize/gpu/swapchain.h>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/image.h>
#include <elasticize/window/window.h>

namespace elastic
{
namespace gpu
{
Swapchain::Swapchain(Engine& engine, const window::Window& window)
  : engine_(engine)
{
  auto instance = engine.instance();
  auto physicalDevice = engine.physicalDevice();
  auto queueIndex = engine.queueIndex();
  auto device = engine.device();

  surface_ = window.createVulkanSurface(instance);

  // Swapchain
  if (!physicalDevice.getSurfaceSupportKHR(queueIndex, surface_))
    throw std::runtime_error("Device queue does not support surface");

  const auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface_);

  // Triple buffering
  auto imageCount = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    imageCount = capabilities.maxImageCount;
  if (imageCount != 3)
    throw std::runtime_error("Triple buffering is not supported");

  // Present mode: use mailbox if available. Limit fps in draw call
  vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
  const auto presentModes = physicalDevice.getSurfacePresentModesKHR(surface_);
  for (auto availableMode : presentModes)
  {
    if (availableMode == vk::PresentModeKHR::eMailbox)
    {
      presentMode = vk::PresentModeKHR::eMailbox;
      break;
    }
  }

  // Swapchain format
  const auto availableFormats = physicalDevice.getSurfaceFormatsKHR(surface_);
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

  swapchain_ = device.createSwapchainKHR(swapchainInfo_);

  for (auto image : device.getSwapchainImagesKHR(swapchain_))
    swapchainImages_.emplace_back(engine_, image, format.format);
}

Swapchain::~Swapchain()
{
  auto instance = engine_.instance();
  auto device = engine_.device();

  swapchainImages_.clear();
  device.destroySwapchainKHR(swapchain_);

  instance.destroySurfaceKHR(surface_);
}

const Image& Swapchain::image(uint32_t index) const
{
  return swapchainImages_[index];
}
}
}
