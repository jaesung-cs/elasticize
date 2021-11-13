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

  auto instance() const noexcept { return instance_; }
  auto physicalDevice() const noexcept { return physicalDevice_; }
  auto queue() const noexcept { return queue_; }
  auto queueIndex() const noexcept { return queueIndex_; }
  auto device() const noexcept { return device_; }
  auto transientCommandPool() const noexcept { return transientCommandPool_; }
  auto descriptorPool() const noexcept { return descriptorPool_; }

private:
  // By friend objects
  vk::Buffer createBuffer(vk::DeviceSize size);

  void bindImageMemory(vk::Image image);

  vk::ShaderModule createShaderModule(const std::string& filepath);

private:
  void createInstance();
  void destroyInstance();

  void selectSuitablePhysicalDevice();
  void createDevice();
  void destroyDevice();

  void createMemoryPool();
  void destroyMemoryPool();

  void createCommandPool();
  void destroyCommandPool();

  void createDescriptorPool();
  void destroyDescriptorPool();

private:
  Options options_;
  vk::Instance instance_;
  vk::DebugUtilsMessengerEXT messenger_;

  vk::PhysicalDevice physicalDevice_;
  vk::Device device_;
  vk::Queue queue_;
  uint32_t queueIndex_ = 0;

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

  // Descriptor pool
  vk::DescriptorPool descriptorPool_;

  // Compute pipelines
  struct ComputePipeline
  {
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
  };
  std::vector<ComputePipeline> computePipelines_;

  // Descriptor sets
  std::vector<vk::DescriptorSet> descriptorSets_;
};
}
}

#endif // ELASTICIZE_GPU_ENGINE_H_
