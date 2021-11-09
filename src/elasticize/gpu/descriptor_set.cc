#include <elasticize/gpu/descriptor_set.h>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/descriptor_set_layout.h>

namespace elastic
{
namespace gpu
{
DescriptorSet::DescriptorSet(Engine& engine,
  DescriptorSetLayout& descriptorSetLayout,
  std::initializer_list<BufferProxy> bufferProxies)
  : engine_(engine)
{
  auto device = engine_.device();
  auto descriptorPool = engine_.descriptorPool();

  std::vector<vk::Buffer> buffers;
  for (auto bufferProxy : bufferProxies)
    buffers.push_back(bufferProxy);

  vk::DescriptorSetLayout setLayout = descriptorSetLayout;

  const auto descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo()
    .setDescriptorPool(descriptorPool)
    .setSetLayouts(setLayout);

  descriptorSet_ = device.allocateDescriptorSets(descriptorSetAllocateInfo)[0];

  std::vector<vk::DescriptorBufferInfo> bufferInfos(buffers.size());
  std::vector<vk::WriteDescriptorSet> writes(buffers.size());
  for (int i = 0; i < buffers.size(); i++)
  {
    bufferInfos[i]
      .setBuffer(buffers[i])
      .setOffset(0)
      .setRange(VK_WHOLE_SIZE);

    writes[i]
      .setDstBinding(i)
      .setDstSet(descriptorSet_)
      .setDescriptorType(vk::DescriptorType::eStorageBuffer)
      .setDescriptorCount(1)
      .setBufferInfo(bufferInfos[i]);
  }

  device.updateDescriptorSets(writes, {});
}

DescriptorSet::~DescriptorSet()
{
  auto device = engine_.device();
  auto descriptorPool = engine_.descriptorPool();

  device.freeDescriptorSets(descriptorPool, descriptorSet_);
}
}
}
