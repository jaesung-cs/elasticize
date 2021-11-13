#include <elasticize/gpu/image.h>

#include <iostream>

#include <elasticize/gpu/engine.h>

namespace elastic
{
namespace gpu
{
Image::Image(Engine& engine, const Options& options)
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

Image::Image(Engine& engine, vk::Image image, vk::Format format)
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

Image::~Image()
{
  auto device = engine_.device();

  if (imageView_)
    device.destroyImageView(imageView_);

  if (image_ && created_)
    device.destroyImage(image_);
}
}
}
