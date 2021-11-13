#ifndef ELASTICIZE_GPU_BUFFER_INL_
#define ELASTICIZE_GPU_BUFFER_INL_

#include <elasticize/gpu/engine.h>

namespace elastic
{
namespace gpu
{
template <typename T>
Buffer<T>::Buffer(Engine& engine, uint64_t count)
  : engine_(engine)
  , data_(count)
{
  buffer_ = engine.createBuffer(sizeof(T) * count);
}

template <typename T>
Buffer<T>::Buffer(Engine& engine, std::initializer_list<T> values)
  : engine_(engine)
  , data_(std::move(values))
{
  buffer_ = engine.createBuffer(sizeof(T) * values.size());
}

template <typename T>
Buffer<T>::~Buffer()
{
  engine_.destroyBuffer(buffer_);
}

template <typename T>
void Buffer<T>::toGpu()
{
  engine_.transferToGpu(data_, buffer_);
}

template <typename T>
void Buffer<T>::fromGpu()
{
  engine_.transferFromGpu(data_, buffer_);
}
}
}

#endif // ELASTICIZE_GPU_BUFFER_INL_
