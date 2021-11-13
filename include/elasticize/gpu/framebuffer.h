#ifndef ELASTICIZE_GPU_FRAMEBUFFER_H_
#define ELASTICIZE_GPU_FRAMEBUFFER_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace gpu
{
class Engine;
class GraphicsShader;
class Image;

class Framebuffer
{
public:
  Framebuffer() = delete;
  Framebuffer(Engine& engine, uint32_t width, uint32_t height, GraphicsShader& graphicsShader, std::initializer_list<std::reference_wrapper<const Image>> attachments);
  ~Framebuffer();

  operator vk::Framebuffer() const noexcept { return framebuffer_; }

  auto width() const noexcept { return width_; }
  auto height() const noexcept { return height_; }

private:
  Engine& engine_;

  uint32_t width_;
  uint32_t height_;
  vk::Framebuffer framebuffer_;
};
}
}

#endif // ELASTICIZE_GPU_FRAMEBUFFER_H_
