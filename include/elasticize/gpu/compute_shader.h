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

  ComputeShader(Engine engine,
    const std::string& filepath,
    DescriptorSetLayout descriptorSetLayout,
    const std::vector<PushConstantRange>& pushConstantRanges);

  ~ComputeShader();

  vk::PipelineLayout pipelineLayout() const noexcept;
  vk::Pipeline pipeline() const noexcept;

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#endif // ELASTICIZE_GPU_COMPUTE_SHADER_H_
