#include <elasticize/gpu/descriptor_set_layout.h>

#include <elasticize/gpu/engine.h>

namespace elastic
{
namespace gpu
{
DescriptorSetLayout::DescriptorSetLayout(Engine& engine, uint32_t storageBufferCount)
  : engine_(engine)
{
  auto device = engine_.device();

  // Descriptor set layout
  std::vector<vk::DescriptorSetLayoutBinding> bindings(storageBufferCount);
  for (int i = 0; i < storageBufferCount; i++)
  {
    bindings[i]
      .setBinding(i)
      .setStageFlags(vk::ShaderStageFlagBits::eCompute)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDescriptorCount(1);
  }

  const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo().setBindings(bindings);
  descriptorSetLayout_ = device.createDescriptorSetLayout(descriptorSetLayoutInfo);
}

DescriptorSetLayout::~DescriptorSetLayout()
{
  auto device = engine_.device();

  device.destroyDescriptorSetLayout(descriptorSetLayout_);
}
}
}
