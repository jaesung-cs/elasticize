#ifndef ELASTICIZE_GPU_SWAPCHAIN_H_
#define ELASTICIZE_GPU_SWAPCHAIN_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace window
{
class Window;
}

namespace gpu
{
class Engine;
class Image;

class Swapchain
{
public:
  Swapchain() = delete;
  Swapchain(Engine& engine, const window::Window& window);
  ~Swapchain();

  const auto& info() const noexcept { return swapchainInfo_; }
  auto imageCount() const noexcept { return swapchainImages_.size(); }
  const auto& images() const noexcept { return swapchainImages_; }
  const Image& image(uint32_t index) const;

private:
  Engine& engine_;

  vk::SurfaceKHR surface_;
  vk::SwapchainCreateInfoKHR swapchainInfo_;
  vk::SwapchainKHR swapchain_;
  std::vector<Image> swapchainImages_;
};
}
}

#endif // ELASTICIZE_GPU_SWAPCHAIN_H_
