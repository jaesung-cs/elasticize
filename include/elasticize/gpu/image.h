#ifndef ELASTICIZE_GPU_IMAGE_H_
#define ELASTICIZE_GPU_IMAGE_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace gpu
{
class Engine;

class Image
{
public:
  struct Options
  {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;
    vk::ImageUsageFlags usage = {};
    vk::Format format = vk::Format::eR8G8B8A8Srgb;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
  };

public:
  Image() = delete;
  Image(Engine& engine, const Options& options);
  Image(Engine& engine, vk::Image image, vk::Format format);
  ~Image();

  Image(const Image& rhs) = delete;
  Image& operator = (const Image& rhs) = delete;

  Image(Image&& rhs) noexcept
    : engine_(rhs.engine_)
    , created_(rhs.created_)
    , image_(rhs.image_)
    , imageView_(rhs.imageView_)
  {
    rhs.image_ = nullptr;
    rhs.imageView_ = nullptr;
  }

  Image& operator = (Image&& rhs) noexcept
  {
    created_ = rhs.created_;
    image_ = rhs.image_;
    imageView_ = rhs.imageView_;

    rhs.image_ = nullptr;
    rhs.imageView_ = nullptr;

    return *this;
  }

  operator vk::Image() const noexcept { return image_; }

  auto imageView() const noexcept { return imageView_; }

private:
  Engine& engine_;
  bool created_;

  vk::Image image_;
  vk::ImageView imageView_;
};
}
}

#endif // ELASTICIZE_GPU_IMAGE_H_
