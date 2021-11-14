#ifndef ELASTICIZE_GPU_GRAPHICS_SHADER_H_
#define ELASTICIZE_GPU_GRAPHICS_SHADER_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace gpu
{
class Engine;
class DescriptorSetLayout;

class GraphicsShader
{
private:
  struct Shader
  {
    vk::ShaderStageFlagBits stage;
    std::string filepath;
  };

  struct Binding
  {
    uint32_t binding;
    uint32_t stride;
  };

  struct Attribute
  {
    uint32_t binding;
    uint32_t location;
    vk::Format format;
    uint32_t offset;
  };

public:
  struct Options
  {
    DescriptorSetLayout* pDescriptorSetLayout = nullptr;
    std::vector<Shader> shaders;
    std::vector<Binding> bindings;
    std::vector<Attribute> attributes;
    vk::Format imageFormat;
  };

public:
  GraphicsShader() = delete;

  GraphicsShader(Engine engine, const Options& options);

  ~GraphicsShader();

  vk::RenderPass renderPass() const noexcept;
  vk::PipelineLayout pipelineLayout() const noexcept;
  vk::Pipeline pipeline() const noexcept;

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#endif // ELASTICIZE_GPU_GRAPHICS_SHADER_H_
