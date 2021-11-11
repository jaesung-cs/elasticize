#ifndef ELASTICIZE_GPU_COMPUTE_SHADER_H_
#define ELASTICIZE_GPU_COMPUTE_SHADER_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace gpu
{
class Engine;
class DescriptorSetLayout;

struct PushConstantRange
{
};

class ComputeShader
{
public:
  ComputeShader() = delete;

  ComputeShader(Engine& engine,
    const std::string& filepath,
    DescriptorSetLayout& descriptorSetLayout,
    const std::vector<PushConstantRange>& pushConstantRanges);

  ~ComputeShader();

  auto pipelineLayout() const noexcept { return pipelineLayout_; }
  auto pipeline() const noexcept { return pipeline_; }

private:
  Engine& engine_;

  vk::PipelineLayout pipelineLayout_;
  vk::Pipeline pipeline_;
};
}
}

#endif // ELASTICIZE_GPU_COMPUTE_SHADER_H_
