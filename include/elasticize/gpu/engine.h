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

    vk::DeviceSize memoryPoolSize = 256ull * 1024 * 1024; // 256MB default
  };

public:
  Engine() = delete;
  explicit Engine(Options options);
  ~Engine();

  void attachWindow(const window::Window& window);

private:
  // By friend objects
  vk::Buffer createBuffer(vk::DeviceSize size);
  void destroyBuffer(vk::Buffer buffer);

  template <typename T>
  void transferToGpu(const std::vector<T>& data, vk::Buffer buffer)
  {
    transferToGpu(data.data(), sizeof(T) * data.size(), buffer);
  }

  void transferToGpu(const void* data, vk::DeviceSize size, vk::Buffer buffer);

  template <typename T>
  void transferFromGpu(std::vector<T>& data, vk::Buffer buffer)
  {
    transferFromGpu(data.data(), sizeof(T) * data.size(), buffer);
  }

  void transferFromGpu(void* data, vk::DeviceSize size, vk::Buffer buffer);

private:
  void createInstance();
  void destroyInstance();

  void selectSuitablePhysicalDevice();
  void createDevice();
  void destroyDevice();

  void createSwapchain(const window::Window& window);
  void destroySwapchain();

  void createMemoryPool();
  void destroyMemoryPool();

  void createCommandPool();
  void destroyCommandPool();

private:
  Options options_;
  vk::Instance instance_;
  vk::DebugUtilsMessengerEXT messenger_;

  vk::PhysicalDevice physicalDevice_;
  vk::Device device_;
  vk::Queue queue_;
  uint32_t queueIndex_;

  // Memory pool
  uint32_t deviceIndex_ = 0;
  uint32_t hostIndex_ = 0;
  vk::DeviceMemory deviceMemory_;
  vk::DeviceSize deviceMemoryOffset_ = 0;
  vk::DeviceMemory hostMemory_;
  vk::Buffer stagingBuffer_;
  uint8_t* stagingBufferMap_ = nullptr;

  // Command pool
  vk::CommandPool transientCommandPool_;
  vk::Fence transferFence_;

  vk::SurfaceKHR surface_;
  vk::SwapchainCreateInfoKHR swapchainInfo_;
  vk::SwapchainKHR swapchain_;
};
}
}

#endif // ELASTICIZE_GPU_ENGINE_H_
