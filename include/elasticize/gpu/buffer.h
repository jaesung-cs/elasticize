#ifndef ELASTICIZE_GPU_BUFFER_H_
#define ELASTICIZE_GPU_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace gpu
{
class Engine;

template <typename T>
class Buffer
{
public:
  Buffer() = delete;
  Buffer(Engine engine, uint64_t count);
  Buffer(Engine engine, std::initializer_list<T> values);
  ~Buffer();

  operator vk::Buffer() const noexcept;

  T& operator [] (uint64_t index);
  const T& operator [] (uint64_t index) const;

  T* data();
  const T* data() const;
  uint64_t size() const;

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}
}

#include <elasticize/gpu/buffer.inl>

#endif // ELASTICIZE_GPU_BUFFER_H_
