#ifndef ELASTICIZE_GPU_BUFFER_INL_
#define ELASTICIZE_GPU_BUFFER_INL_

#include <elasticize/gpu/engine.h>

namespace elastic
{
namespace gpu
{
template <typename T>
class Buffer<T>::Impl
{
public:
  Impl() = delete;

  Impl(Engine engine, uint64_t count)
    : engine_(engine)
    , data_(count)
  {
    buffer_ = engine.createBuffer(sizeof(T) * count);
  }

  Impl(Engine engine, std::initializer_list<T> values)
    : engine_(engine)
    , data_(std::move(values))
  {
    buffer_ = engine.createBuffer(sizeof(T) * values.size());
  }

  ~Impl()
  {
    auto device = engine_.device();
    device.destroyBuffer(buffer_);
  }

  operator vk::Buffer() const noexcept { return buffer_; }

  auto& operator [] (uint64_t index) { return data_[index]; }
  const auto& operator [] (uint64_t index) const { return data_[index]; }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  auto size() const { return data_.size(); }

private:
  Engine engine_;

  std::vector<T> data_;
  vk::Buffer buffer_;
};

template <typename T>
Buffer<T>::Buffer(Engine engine, uint64_t count)
  : impl_(std::make_shared<Impl>(engine, count))
{
}

template <typename T>
Buffer<T>::Buffer(Engine engine, std::initializer_list<T> values)
  : impl_(std::make_shared<Impl>(engine, std::move(values)))
{
}

template <typename T>
Buffer<T>::~Buffer() = default;

template <typename T>
Buffer<T>::operator vk::Buffer() const noexcept { return *impl_; }

template <typename T>
T& Buffer<T>::operator [] (uint64_t index) { return (*impl_)[index]; }

template <typename T>
const T& Buffer<T>::operator [] (uint64_t index) const { return (*impl_)[index]; }

template <typename T>
T* Buffer<T>::data() { return impl_->data(); }

template <typename T>
const T* Buffer<T>::data() const { return impl_->data(); }

template <typename T>
uint64_t Buffer<T>::size() const { return impl_->size(); }
}
}

#endif // ELASTICIZE_GPU_BUFFER_INL_
