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
class GraphicsShader;
class DescriptorSet;
class Framebuffer;

class Execution
{
public:
  Execution() = delete;
  Execution(Engine& engine);
  ~Execution();

  Execution(const Execution& rhs) = delete;
  Execution& operator = (const Execution& rhs) = delete;

  Execution(Execution&& rhs) noexcept
    : engine_(rhs.engine_)
  {
    fence_ = rhs.fence_;
    commandBuffer_ = rhs.commandBuffer_;
    stagingBufferOffsetToGpu_ = rhs.stagingBufferOffsetToGpu_;
    fromGpus_ = std::move(rhs.fromGpus_);
    stagingBufferOffsetFromGpu_ = rhs.stagingBufferOffsetFromGpu_;

    rhs.fence_ = nullptr;
    rhs.commandBuffer_ = nullptr;
    rhs.stagingBufferOffsetToGpu_ = 0;
    rhs.fromGpus_.clear();
    rhs.stagingBufferOffsetFromGpu_ = 0;
  }

  Execution& operator = (Execution&& rhs) noexcept
  {
    fence_ = rhs.fence_;
    commandBuffer_ = rhs.commandBuffer_;
    stagingBufferOffsetToGpu_ = rhs.stagingBufferOffsetToGpu_;
    fromGpus_ = std::move(rhs.fromGpus_);
    stagingBufferOffsetFromGpu_ = rhs.stagingBufferOffsetFromGpu_;

    rhs.fence_ = nullptr;
    rhs.commandBuffer_ = nullptr;
    rhs.stagingBufferOffsetToGpu_ = 0;
    rhs.fromGpus_.clear();
    rhs.stagingBufferOffsetFromGpu_ = 0;

    return *this;
  }

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

  template <typename T>
  Execution& draw(GraphicsShader& graphicsShader, DescriptorSet& descriptorSet, Framebuffer& framebuffer, const Buffer<T>& vertexBuffer, const Buffer<uint32_t>& indexBuffer)
  {
    return draw(graphicsShader, descriptorSet, framebuffer, vertexBuffer, indexBuffer, static_cast<uint32_t>(indexBuffer.size()));
  }

  void run();

private:
  Execution& toGpu(vk::Buffer buffer, const void* data, vk::DeviceSize size);
  Execution& fromGpu(vk::Buffer buffer, void* data, vk::DeviceSize size);
  Execution& copy(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
  Execution& runComputeShader(ComputeShader& computeShader, DescriptorSet& descriptorSet, uint32_t groupCountX, const void* pushConstants, uint32_t size);
  Execution& draw(GraphicsShader& graphicsShader, DescriptorSet& descriptorSet, Framebuffer& framebuffer, vk::Buffer vertexBuffer, vk::Buffer indexBuffer, uint32_t indexCount);

  Engine& engine_;

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
