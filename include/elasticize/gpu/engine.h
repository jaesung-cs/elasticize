#ifndef ELASTICIZE_GPU_ENGINE_H_
#define ELASTICIZE_GPU_ENGINE_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace window
{
class Window;
}

namespace gpu
{
template <typename T>
class Buffer;

class Engine
{
  template <typename T>
  friend class Buffer;

public:
  struct Options
  {
    bool validationLayer = false;
    bool headless = true;
  };

public:
  Engine() = delete;
  explicit Engine(Options options);
  ~Engine();

  void attachWindow(const window::Window& window);

private:
  // By friend objects
  vk::Buffer createBuffer(uint64_t size);
  void destroyBuffer(vk::Buffer buffer);

private:
  void createInstance();
  void destroyInstance();

  void selectSuitablePhysicalDevice();
  void createDevice();
  void destroyDevice();

  void createSwapchain(const window::Window& window);
  void destroySwapchain();

private:
  Options options_;
  vk::Instance instance_;
  vk::DebugUtilsMessengerEXT messenger_;

  vk::PhysicalDevice physicalDevice_;
  vk::Device device_;
  vk::Queue queue_;
  uint32_t queueIndex_;

  vk::SurfaceKHR surface_;
  vk::SwapchainCreateInfoKHR swapchainInfo_;
  vk::SwapchainKHR swapchain_;
};
}
}

#endif // ELASTICIZE_GPU_ENGINE_H_
