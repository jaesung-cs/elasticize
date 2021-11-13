#include <elasticize/gpu/compute_shader.h>

#include <fstream>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/descriptor_set_layout.h>

namespace elastic
{
namespace gpu
{
ComputeShader::ComputeShader(Engine& engine,
  const std::string& filepath,
  DescriptorSetLayout& descriptorSetLayout,
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

ComputeShader::~ComputeShader()
{
  auto device = engine_.device();

  device.destroyPipeline(pipeline_);
  device.destroyPipelineLayout(pipelineLayout_);
}
}
}
