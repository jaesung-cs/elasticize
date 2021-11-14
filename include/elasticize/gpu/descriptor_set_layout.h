#ifndef ELASTICIZE_GPU_DESCRIPTOR_SET_LAYOUT_H_
#define ELASTICIZE_GPU_DESCRIPTOR_SET_LAYOUT_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace gpu
{
class Engine;

class DescriptorSetLayout
{
public:
  DescriptorSetLayout() = delete;
  DescriptorSetLayout(Engine engine, uint32_t storageBufferCount);
  ~DescriptorSetLayout();

  operator vk::DescriptorSetLayout() const noexcept;

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#endif // ELASTICIZE_GPU_DESCRIPTOR_SET_LAYOUT_H_
