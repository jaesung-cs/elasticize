#include <elasticize/gpu/framebuffer.h>

#include <elasticize/gpu/engine.h>

#include <elasticize/gpu/graphics_shader.h>
#include <elasticize/gpu/image.h>

namespace elastic
{
namespace gpu
{
class Framebuffer::Impl
{
public:
  Impl() = delete;

  Impl(Engine engine, uint32_t width, uint32_t height, GraphicsShader graphicsShader, std::initializer_list<std::reference_wrapper<const Image>> attachments)
    : engine_(engine)
    , width_(width)
    , height_(height)
  {
    auto device = engine_.device();

    std::vector<vk::ImageView> imageViews;
    for (const auto& attachment : attachments)
      imageViews.push_back(attachment.get().imageView());

    auto framebufferInfo = vk::FramebufferCreateInfo()
      .setWidth(width)
      .setHeight(height)
      .setRenderPass(graphicsShader.renderPass())
      .setAttachments(imageViews)
      .setLayers(1);

    framebuffer_ = device.createFramebuffer(framebufferInfo);
  }

  ~Impl()
  {
    auto device = engine_.device();

    if (framebuffer_)
      device.destroyFramebuffer(framebuffer_);
  }

  operator vk::Framebuffer() const noexcept { return framebuffer_; }

  auto width() const noexcept { return width_; }
  auto height() const noexcept { return height_; }

private:
  Engine engine_;

  uint32_t width_;
  uint32_t height_;
  vk::Framebuffer framebuffer_;
};

Framebuffer::Framebuffer(Engine engine, uint32_t width, uint32_t height, GraphicsShader graphicsShader, std::initializer_list<std::reference_wrapper<const Image>> attachments)
  : impl_(std::make_shared<Impl>(engine, width, height, graphicsShader, attachments))
{
}

Framebuffer::~Framebuffer() = default;

Framebuffer::operator vk::Framebuffer() const noexcept { return *impl_; }

uint32_t Framebuffer::width() const noexcept { return impl_->width(); }
uint32_t Framebuffer::height() const noexcept { return impl_->height(); }
}
}
