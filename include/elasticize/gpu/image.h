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
  Image(Engine engine, const Options& options);
  Image(Engine engine, vk::Image image, vk::Format format);
  ~Image();

  operator vk::Image() const noexcept;

  vk::ImageView imageView() const noexcept;

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#endif // ELASTICIZE_GPU_IMAGE_H_
