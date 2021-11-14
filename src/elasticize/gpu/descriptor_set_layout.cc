#include <elasticize/gpu/descriptor_set_layout.h>

#include <elasticize/gpu/engine.h>

namespace elastic
{
namespace gpu
{
class DescriptorSetLayout::Impl
{
public:
  Impl() = delete;

  Impl(Engine engine, uint32_t storageBufferCount)
    : engine_(engine)
  {
    auto device = engine_.device();

    // Descriptor set layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings(storageBufferCount);
    for (uint32_t i = 0; i < storageBufferCount; i++)
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

  ~Impl()
  {
    auto device = engine_.device();

    device.destroyDescriptorSetLayout(descriptorSetLayout_);
  }

  operator vk::DescriptorSetLayout() const noexcept { return descriptorSetLayout_; }

private:
  Engine engine_;

  vk::DescriptorSetLayout descriptorSetLayout_;
};

DescriptorSetLayout::DescriptorSetLayout(Engine engine, uint32_t storageBufferCount)
  : impl_(std::make_shared<Impl>(engine, storageBufferCount))
{
}

DescriptorSetLayout::~DescriptorSetLayout() = default;

DescriptorSetLayout::operator vk::DescriptorSetLayout() const noexcept
{
  return *impl_;
}
}
}
