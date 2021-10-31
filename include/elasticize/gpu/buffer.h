#ifndef ELASTICIZE_GPU_BUFFER_H_
#define ELASTICIZE_GPU_BUFFER_H_

#include <vulkan/vulkan.hpp>

namespace elastic
{
namespace gpu
{
template <typename T>
class Buffer
{
  friend class Engine;

public:
  Buffer() = delete;
  Buffer(Engine& engine, uint64_t count);
  ~Buffer();

  auto& operator [] (uint64_t index) { return data_[index]; }
  const auto& operator [] (uint64_t index) const { return data_[index]; }

  void toGpu();
  void fromGpu();

private:
  auto buffer() const { return buffer_; }

private:
  Engine& engine_;

  std::vector<T> data_;
  vk::Buffer buffer_;
};
}
}

#include <elasticize/gpu/buffer.inl>

#endif // ELASTICIZE_GPU_BUFFER_H_
