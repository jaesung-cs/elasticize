#include <elasticize/gpu/framebuffer.h>

#include <elasticize/gpu/engine.h>

#include <elasticize/gpu/graphics_shader.h>
#include <elasticize/gpu/image.h>

namespace elastic
{
namespace gpu
{
Framebuffer::Framebuffer(Engine& engine, uint32_t width, uint32_t height, GraphicsShader& graphicsShader, std::initializer_list<std::reference_wrapper<const Image>> attachments)
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

Framebuffer::~Framebuffer()
{
  auto device = engine_.device();

  device.destroyFramebuffer(framebuffer_);
}
}
}
