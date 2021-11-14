#include <elasticize/gpu/execution.h>

#include <iostream>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/compute_shader.h>
#include <elasticize/gpu/graphics_shader.h>
#include <elasticize/gpu/descriptor_set.h>
#include <elasticize/gpu/framebuffer.h>
#include <elasticize/gpu/swapchain.h>

namespace elastic
{
namespace gpu
{
class Execution::Impl
{
public:
  Impl() = delete;

  Impl(Engine engine)
    : engine_(engine)
  {
    auto device = engine_.device();
    auto transientCommandPool = engine_.transientCommandPool();

    fence_ = device.createFence({});

    const auto allocateInfo = vk::CommandBufferAllocateInfo()
      .setLevel(vk::CommandBufferLevel::ePrimary)
      .setCommandPool(transientCommandPool)
      .setCommandBufferCount(1);
    commandBuffer_ = device.allocateCommandBuffers(allocateInfo)[0];

    commandBuffer_.begin(vk::CommandBufferBeginInfo());
  }

  ~Impl()
  {
    auto device = engine_.device();
    auto transientCommandPool = engine_.transientCommandPool();

    device.freeCommandBuffers(transientCommandPool, commandBuffer_);
    device.destroyFence(fence_);
  }

  void toGpu(vk::Buffer buffer, const void* data, vk::DeviceSize size)
  {
    auto stagingBuffer = engine_.stagingBuffer();

    // Copy data to staging buffer
    engine_.toStagingBuffer(stagingBufferOffsetToGpu_, data, size);

    // Copy command
    auto region = vk::BufferCopy()
      .setSrcOffset(stagingBufferOffsetToGpu_)
      .setDstOffset(0)
      .setSize(size);

    commandBuffer_.copyBuffer(stagingBuffer, buffer, region);

    stagingBufferOffsetToGpu_ += size;
  }

  void fromGpu(vk::Buffer buffer, void* data, vk::DeviceSize size)
  {
    auto stagingBuffer = engine_.stagingBuffer();

    // Record memcpy targets
    fromGpus_.push_back({ data, stagingBufferOffsetFromGpu_, size });

    auto region = vk::BufferCopy()
      .setSrcOffset(0)
      .setDstOffset(stagingBufferOffsetFromGpu_)
      .setSize(size);

    commandBuffer_.copyBuffer(buffer, stagingBuffer, region);

    stagingBufferOffsetFromGpu_ += size;
  }

  void copy(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
  {
    auto region = vk::BufferCopy()
      .setSrcOffset(0)
      .setDstOffset(0)
      .setSize(size);
    commandBuffer_.copyBuffer(srcBuffer, dstBuffer, region);
  }

  void runComputeShader(ComputeShader computeShader, DescriptorSet descriptorSet, uint32_t groupCountX, const void* pushConstants, uint32_t size)
  {
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeShader.pipelineLayout(), 0u, static_cast<vk::DescriptorSet>(descriptorSet), {});
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, computeShader.pipeline());
    commandBuffer_.pushConstants(computeShader.pipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0u, size, pushConstants);
    commandBuffer_.dispatch(groupCountX, 1, 1);
  }

  void barrier()
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
  }

  void draw(GraphicsShader graphicsShader, DescriptorSet descriptorSet, Framebuffer framebuffer, vk::Buffer vertexBuffer, vk::Buffer indexBuffer, uint32_t indexCount)
  {
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsShader.pipeline());

    const auto width = framebuffer.width();
    const auto height = framebuffer.height();

    auto viewport = vk::Viewport()
      .setX(0.f)
      .setY(0.f)
      .setWidth(static_cast<float>(width))
      .setHeight(static_cast<float>(height))
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

  }

  void end()
  {
    commandBuffer_.end();
  }

  void run()
  {
    auto device = engine_.device();
    auto queue = engine_.queue();

    const auto submit = vk::SubmitInfo().setCommandBuffers(commandBuffer_);
    queue.submit(submit, fence_);

    device.waitForFences(fence_, true, UINT64_MAX);
    device.resetFences(fence_);
    commandBuffer_.reset();

    // From staging buffer to actual buffers
    for (auto fromGpuRange : fromGpus_)
      engine_.fromStagingBuffer(fromGpuRange.target, fromGpuRange.stagingBufferOffset, fromGpuRange.size);
  }

  void present(vk::Semaphore imageAvailableSemaphore, vk::Semaphore renderFinishedSemaphore, vk::Fence renderFinishedFence, Swapchain swapchain, uint32_t imageIndex)
  {
    auto queue = engine_.queue();

    std::vector<vk::Semaphore> waitSemaphores = {
      imageAvailableSemaphore,
    };
    std::vector<vk::PipelineStageFlags> waitMasks = {
      vk::PipelineStageFlagBits::eColorAttachmentOutput
    };

    auto submitInfo = vk::SubmitInfo()
      .setWaitSemaphores(waitSemaphores)
      .setWaitDstStageMask(waitMasks)
      .setCommandBuffers(commandBuffer_)
      .setSignalSemaphores(renderFinishedSemaphore);
    queue.submit(submitInfo, renderFinishedFence);

    // Present
    std::vector<vk::SwapchainKHR> swapchains = {
      swapchain,
    };
    auto presentInfo = vk::PresentInfoKHR()
      .setWaitSemaphores(renderFinishedSemaphore)
      .setSwapchains(swapchains)
      .setImageIndices(imageIndex);
    queue.presentKHR(presentInfo);
  }

private:
  Engine engine_;

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

Execution::Execution(Engine engine)
  : impl_(std::make_shared<Impl>(engine))
{
}

Execution::~Execution() = default;

Execution& Execution::toGpu(vk::Buffer buffer, const void* data, vk::DeviceSize size)
{
  impl_->toGpu(buffer, data, size);
  return *this;
}

Execution& Execution::fromGpu(vk::Buffer buffer, void* data, vk::DeviceSize size)
{
  impl_->fromGpu(buffer, data, size);
  return *this;
}

Execution& Execution::copy(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
  impl_->copy(srcBuffer, dstBuffer, size);
  return *this;
}

Execution& Execution::runComputeShader(ComputeShader computeShader, DescriptorSet descriptorSet, uint32_t groupCountX, const void* pushConstants, uint32_t size)
{
  impl_->runComputeShader(computeShader, descriptorSet, groupCountX, pushConstants, size);
  return *this;
}

Execution& Execution::barrier()
{
  impl_->barrier();
  return *this;
}

Execution& Execution::draw(GraphicsShader graphicsShader, DescriptorSet descriptorSet, Framebuffer framebuffer, vk::Buffer vertexBuffer, vk::Buffer indexBuffer, uint32_t indexCount)
{
  impl_->draw(graphicsShader, descriptorSet, framebuffer, vertexBuffer, indexBuffer, indexCount);
  return *this;
}

Execution& Execution::end()
{
  impl_->end();
  return *this;
}

void Execution::run()
{
  impl_->run();
}

void Execution::present(vk::Semaphore imageAvailableSemaphore, vk::Semaphore renderFinishedSemaphore, vk::Fence renderFinishedFence, Swapchain swapchain, uint32_t imageIndex)
{
  impl_->present(imageAvailableSemaphore, renderFinishedSemaphore, renderFinishedFence, swapchain, imageIndex);
}
}
}
