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
  Swapchain(Engine engine, const window::Window& window);
  ~Swapchain();

  operator vk::SwapchainKHR() const noexcept;
  const vk::SwapchainCreateInfoKHR& info() const noexcept;
  uint32_t imageCount() const noexcept;
  const std::vector<Image>& images() const noexcept;
  Image image(uint32_t index) const;

  uint32_t acquireNextImage(vk::Semaphore signalSemaphore);

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#endif // ELASTICIZE_GPU_SWAPCHAIN_H_
