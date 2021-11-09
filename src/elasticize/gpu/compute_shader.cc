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
  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filepath);

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  std::vector<uint32_t> code;
  auto* intPtr = reinterpret_cast<uint32_t*>(buffer.data());
  for (int i = 0; i < fileSize / 4; i++)
    code.push_back(intPtr[i]);

  const auto shaderModuleInfo = vk::ShaderModuleCreateInfo().setCode(code);
  const auto module = device.createShaderModule(shaderModuleInfo);

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
