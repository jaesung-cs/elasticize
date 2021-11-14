#include <elasticize/gpu/image.h>

#include <iostream>

#include <elasticize/gpu/engine.h>

namespace elastic
{
namespace gpu
{
class Image::Impl
{
public:
  Impl() = delete;

  Impl(Engine engine, const Options& options)
    : engine_(engine)
    , created_(true)
  {
    auto device = engine_.device();

    auto imageInfo = vk::ImageCreateInfo()
      .setImageType(vk::ImageType::e2D)
      .setTiling(vk::ImageTiling::eOptimal)
      .setInitialLayout(vk::ImageLayout::eUndefined)
      .setUsage(options.usage)
      .setArrayLayers(1)
      .setExtent({ options.width, options.height, 1 })
      .setFormat(options.format)
      .setSamples(options.samples)
      .setMipLevels(options.mipLevels);
    image_ = device.createImage(imageInfo);

    engine_.bindImageMemory(image_);

    const auto aspect = options.format == vk::Format::eD24UnormS8Uint ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
    auto imageViewInfo = vk::ImageViewCreateInfo()
      .setImage(image_)
      .setViewType(vk::ImageViewType::e2D)
      .setComponents({})
      .setFormat(imageInfo.format)
      .setSubresourceRange({ aspect, 0, 1, 0, 1 });
    imageView_ = device.createImageView(imageViewInfo);
  }

  Impl(Engine engine, vk::Image image, vk::Format format)
    : engine_(engine)
    , image_(image)
    , created_(false)
  {
    auto device = engine_.device();

    auto imageViewInfo = vk::ImageViewCreateInfo()
      .setImage(image_)
      .setViewType(vk::ImageViewType::e2D)
      .setComponents({})
      .setFormat(format)
      .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

    imageView_ = device.createImageView(imageViewInfo);
  }

  ~Impl()
  {
    auto device = engine_.device();

    device.destroyImageView(imageView_);

    if (created_)
      device.destroyImage(image_);
  }

  operator vk::Image() const noexcept { return image_; }

  auto imageView() const noexcept { return imageView_; }

private:
  Engine engine_;
  bool created_;

  vk::Image image_;
  vk::ImageView imageView_;
};

Image::Image(Engine engine, const Options& options)
  : impl_(std::make_shared<Impl>(engine, options))
{
}

Image::Image(Engine engine, vk::Image image, vk::Format format)
  : impl_(std::make_shared<Impl>(engine, image, format))
{
}

Image::~Image() = default;

Image::operator vk::Image() const noexcept { return *impl_; }

vk::ImageView Image::imageView() const noexcept { return impl_->imageView(); }
}
}
