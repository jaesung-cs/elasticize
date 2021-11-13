#include <elasticize/gpu/execution.h>

#include <iostream>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/compute_shader.h>
#include <elasticize/gpu/graphics_shader.h>
#include <elasticize/gpu/descriptor_set.h>
#include <elasticize/gpu/framebuffer.h>

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

Execution& Execution::draw(GraphicsShader& graphicsShader, DescriptorSet& descriptorSet, Framebuffer& framebuffer, vk::Buffer vertexBuffer, vk::Buffer indexBuffer, uint32_t indexCount)
{
  commandBuffer_.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsShader.pipeline());

  const auto width = framebuffer.width();
  const auto height = framebuffer.height();

  auto viewport = vk::Viewport()
    .setX(0.f)
    .setY(0.f)
    .setWidth(width)
    .setHeight(height)
    .setMinDepth(0.f)
    .setMaxDepth(1.f);
  commandBuffer_.setViewport(0, viewport);

  auto scissors = vk::Rect2D()
    .setOffset({ 0, 0 })
    .setExtent({ width, height });
  commandBuffer_.setScissor(0, scissors);

  commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsShader.pipelineLayout(), 0, { descriptorSet }, {});

  auto renderArea = vk::Rect2D()
    .setOffset(vk::Offset2D(0, 0))
    .setExtent(vk::Extent2D(framebuffer.width(), framebuffer.height()));

  std::vector<vk::ClearValue> clearValues = {
    vk::ClearValue().setColor(std::array<float, 4>{ 0.f, 0.f, 0.f, 1.f }),
    vk::ClearValue().setDepthStencil(vk::ClearDepthStencilValue(0.f)),
  };

  auto beginInfo = vk::RenderPassBeginInfo()
    .setRenderPass(graphicsShader.renderPass())
    .setRenderArea(renderArea)
    .setClearValues(clearValues)
    .setFramebuffer(framebuffer);

  commandBuffer_.beginRenderPass(beginInfo, vk::SubpassContents::eInline);

  commandBuffer_.bindVertexBuffers(0, { vertexBuffer }, { 0 });
  commandBuffer_.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
  commandBuffer_.drawIndexed(indexCount, 1, 0, 0, 0);

  commandBuffer_.endRenderPass();

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
