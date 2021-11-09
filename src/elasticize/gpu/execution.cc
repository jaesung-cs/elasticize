#include <elasticize/gpu/execution.h>

#include <iostream>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/compute_shader.h>
#include <elasticize/gpu/descriptor_set.h>

namespace elastic
{
namespace gpu
{
Execution::Execution(Engine& engine)
  : engine_(engine)
{
  queue_ = engine.queue();
  device_ = engine.device();
  transientCommandPool_ = engine.transientCommandPool();

  fence_ = device_.createFence({});

  const auto allocateInfo = vk::CommandBufferAllocateInfo()
    .setLevel(vk::CommandBufferLevel::ePrimary)
    .setCommandPool(transientCommandPool_)
    .setCommandBufferCount(1);
  commandBuffer_ = device_.allocateCommandBuffers(allocateInfo)[0];

  commandBuffer_.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
}

Execution::~Execution()
{
  device_.freeCommandBuffers(transientCommandPool_, commandBuffer_);
  device_.destroyFence(fence_);
}

Execution& Execution::toGpu(vk::Buffer buffer, const void* data, vk::DeviceSize size)
{
  auto stagingBuffer = engine_.stagingBuffer_;

  // Copy data to staging buffer
  std::memcpy(engine_.stagingBufferMap_ + stagingBufferOffsetToGpu_, data, size);

  // Copy command
  auto region = vk::BufferCopy()
    .setSrcOffset(stagingBufferOffsetToGpu_)
    .setDstOffset(0)
    .setSize(size);

  commandBuffer_.copyBuffer(stagingBuffer, buffer, region);

  stagingBufferOffsetToGpu_ += size;

  return *this;
}

Execution& Execution::fromGpu(vk::Buffer buffer, void* data, vk::DeviceSize size)
{
  auto stagingBuffer = engine_.stagingBuffer_;

  // Record memcpy targets
  fromGpus_.push_back({ data, stagingBufferOffsetFromGpu_, size });

  auto region = vk::BufferCopy()
    .setSrcOffset(0)
    .setDstOffset(stagingBufferOffsetFromGpu_)
    .setSize(size);

  commandBuffer_.copyBuffer(buffer, stagingBuffer, region);

  stagingBufferOffsetFromGpu_ += size;

  return *this;
}

Execution& Execution::copy(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
  auto region = vk::BufferCopy()
    .setSrcOffset(0)
    .setDstOffset(0)
    .setSize(size);

  commandBuffer_.copyBuffer(srcBuffer, dstBuffer, region);

  return *this;
}

Execution& Execution::runComputeShader(ComputeShader& computeShader, DescriptorSet& descriptorSet, uint32_t groupCountX, const void* pushConstants, uint32_t size)
{
  commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeShader.pipelineLayout(), 0u, static_cast<vk::DescriptorSet>(descriptorSet), {});
  commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, computeShader.pipeline());
  commandBuffer_.pushConstants(computeShader.pipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0u, size, pushConstants);
  commandBuffer_.dispatch(groupCountX, 1, 1);

  return *this;
}

Execution& Execution::barrier()
{
  // TODO: separate shader and transfer read/write?
  const auto memoryBarrier = vk::MemoryBarrier()
    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite)
    .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead);

  commandBuffer_.pipelineBarrier(
    vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
    vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
    {},
    memoryBarrier, {}, {});

  return *this;
}

void Execution::run()
{
  commandBuffer_.end();

  const auto submit = vk::SubmitInfo().setCommandBuffers(commandBuffer_);
  queue_.submit(submit, fence_);

  device_.waitForFences(fence_, true, UINT64_MAX);
  device_.resetFences(fence_);
  commandBuffer_.reset();

  // From staging buffer to actual buffers
  for (auto fromGpuRange : fromGpus_)
    std::memcpy(fromGpuRange.target, engine_.stagingBufferMap_ + fromGpuRange.stagingBufferOffset, fromGpuRange.size);
}
}
}
