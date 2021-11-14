#include <elasticize/gpu/compute_shader.h>

#include <fstream>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/descriptor_set_layout.h>

namespace elastic
{
namespace gpu
{
class ComputeShader::Impl
{
public:
  Impl() = delete;

  Impl(Engine engine,
    const std::string& filepath,
    DescriptorSetLayout descriptorSetLayout,
    const std::vector<PushConstantRange>& pushConstantRanges)
    : engine_(engine)
  {
    auto device = engine_.device();

    // Pipeline layout
    // TODO: push constants
    std::vector<vk::PushConstantRange> pushConstantRange(1);
    pushConstantRange[0]
      .setStageFlags(vk::ShaderStageFlagBits::eCompute)
      .setOffset(0)
      .setSize(sizeof(uint32_t) * 3);

    vk::DescriptorSetLayout setLayout = descriptorSetLayout;

    const auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
      .setSetLayouts(setLayout)
      .setPushConstantRanges(pushConstantRange);

    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);

    // Pipeline
    const auto module = engine.createShaderModule(filepath);

    const auto stage = vk::PipelineShaderStageCreateInfo()
      .setStage(vk::ShaderStageFlagBits::eCompute)
      .setModule(module)
      .setPName("main");

    const auto pipelineInfo = vk::ComputePipelineCreateInfo()
      .setLayout(pipelineLayout_)
      .setStage(stage);

    pipeline_ = device.createComputePipeline(nullptr, pipelineInfo).value;

    device.destroyShaderModule(module);
  }

  ~Impl()
  {
    auto device = engine_.device();

    device.destroyPipeline(pipeline_);
    device.destroyPipelineLayout(pipelineLayout_);
  }

  auto pipelineLayout() const noexcept { return pipelineLayout_; }
  auto pipeline() const noexcept { return pipeline_; }

private:
  Engine engine_;

  vk::PipelineLayout pipelineLayout_;
  vk::Pipeline pipeline_;
};

ComputeShader::ComputeShader(Engine engine,
  const std::string& filepath,
  DescriptorSetLayout descriptorSetLayout,
  const std::vector<PushConstantRange>& pushConstantRanges)
  : impl_(std::make_shared<Impl>(engine, filepath, descriptorSetLayout, pushConstantRanges))
{
}

ComputeShader::~ComputeShader() = default;

vk::PipelineLayout ComputeShader::pipelineLayout() const noexcept
{
  return impl_->pipelineLayout();
}

vk::Pipeline ComputeShader::pipeline() const noexcept
{
  return impl_->pipeline();
}

}
}
