#ifndef ELASTICIZE_GPU_DESCRIPTOR_SET_H_
#define ELASTICIZE_GPU_DESCRIPTOR_SET_H_

#include <vulkan/vulkan.hpp>

#include <elasticize/gpu/buffer.h>

namespace elastic
{
namespace gpu
{
class Engine;
class DescriptorSetLayout;

class DescriptorSet
{
private:
  class BufferProxy
  {
  public:
    BufferProxy() = delete;

    template <typename T>
    BufferProxy(const Buffer<T>& buffer)
      : buffer_(buffer) {}

    operator vk::Buffer() const noexcept { return buffer_; }

  private:
    vk::Buffer buffer_;
  };

public:
  DescriptorSet() = delete;
  DescriptorSet(Engine engine, DescriptorSetLayout descriptorSetLayout, std::initializer_list<BufferProxy> bufferProxies);
  ~DescriptorSet();

  operator vk::DescriptorSet() const noexcept;

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#endif // ELASTICIZE_GPU_DESCRIPTOR_SET_H_
