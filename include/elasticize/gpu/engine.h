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

class Image;
class Execution;
class GraphicsShader;
class ComputeShader;

class Engine
{
  template <typename T>
  friend class Buffer;

  friend class Image;
  friend class Execution;
  friend class GraphicsShader;
  friend class ComputeShader;

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

  vk::Instance instance() const noexcept;
  vk::PhysicalDevice physicalDevice() const noexcept;
  vk::Queue queue() const noexcept;
  uint32_t queueIndex() const noexcept;
  vk::Device device() const noexcept;
  vk::CommandPool transientCommandPool() const noexcept;
  vk::DescriptorPool descriptorPool() const noexcept;

private:
  // By friend objects
  vk::Buffer createBuffer(vk::DeviceSize size);

  void bindImageMemory(vk::Image image);

  vk::ShaderModule createShaderModule(const std::string& filepath);

  vk::Buffer stagingBuffer() const noexcept;
  void fromStagingBuffer(void* target, vk::DeviceSize srcOffset, vk::DeviceSize size);
  void toStagingBuffer(vk::DeviceSize targetOffset, const void* data, vk::DeviceSize size);

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#endif // ELASTICIZE_GPU_ENGINE_H_
