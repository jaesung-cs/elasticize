#ifndef ELASTICIZE_GPU_EXECUTION_H_
#define ELASTICIZE_GPU_EXECUTION_H_

#include <vulkan/vulkan.hpp>

#include <elasticize/gpu/buffer.h>

namespace elastic
{
namespace gpu
{
class Engine;
class ComputeShader;
class DescriptorSet;

class Execution
{
public:
  Execution() = delete;
  Execution(Engine& engine);
  ~Execution();

  template <typename T>
  Execution& toGpu(const Buffer<T>& buffer)
  {
    toGpu(buffer, buffer.data(), sizeof(T) * buffer.size());
    return *this;
  }

  template <typename T>
  Execution& fromGpu(Buffer<T>& buffer)
  {
    fromGpu(buffer, buffer.data(), sizeof(T) * buffer.size());
    return *this;
  }

  template <typename T>
  Execution& copy(const Buffer<T>& srcBuffer, const Buffer<T>& dstBuffer)
  {
    return copy(srcBuffer, dstBuffer, sizeof(T) * srcBuffer.size());
  }

  template <typename T>
  Execution& runComputeShader(ComputeShader& computeShader, DescriptorSet& descriptorSet, uint32_t groupCountX,
    const T& pushConstants)
  {
    return runComputeShader(computeShader, descriptorSet, groupCountX, &pushConstants, sizeof(T));
  }

  Execution& barrier();

  void run();

private:
  Execution& toGpu(vk::Buffer buffer, const void* data, vk::DeviceSize size);
  Execution& fromGpu(vk::Buffer buffer, void* data, vk::DeviceSize size);
  Execution& copy(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
  Execution& runComputeShader(ComputeShader& computeShader, DescriptorSet& descriptorSet, uint32_t groupCountX, const void* pushConstants, uint32_t size);

  Engine& engine_;
  vk::Device device_;
  vk::Queue queue_;
  vk::CommandPool transientCommandPool_;

  vk::Fence fence_;
  vk::CommandBuffer commandBuffer_;
  vk::DeviceSize stagingBufferOffsetToGpu_ = 0;

  struct FromGpu
  {
    void* target;
    vk::DeviceSize stagingBufferOffset;
    vk::DeviceSize size;
  };
  std::vector<FromGpu> fromGpus_;
  vk::DeviceSize stagingBufferOffsetFromGpu_ = 0;
};
}
}

#endif // ELASTICIZE_GPU_EXECUTION_H_
