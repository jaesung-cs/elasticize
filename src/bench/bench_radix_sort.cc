#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/buffer.h>

#include <iostream>
#include <random>

int main()
{
  try
  {
    elastic::gpu::Engine::Options options;
    options.headless = true;
    options.validationLayer = true;
    elastic::gpu::Engine engine(options);

    std::cout << "Engine started!" << std::endl;

    constexpr int n = 1000000;
    constexpr int keyBits = 30; // for 10-bit each component morton code

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> distribution(0, (1 << keyBits) - 1);

    std::vector<uint32_t> buffer(n);
    for (int i = 0; i < n; i++)
      buffer[i] = distribution(gen);

    elastic::gpu::Buffer<uint32_t> gpuBuffer(engine, n);
    for (int i = 0; i < n; i++)
      gpuBuffer[i] = buffer[i];

    gpuBuffer.toGpu();

    for (int i = 0; i < n; i++)
      gpuBuffer[i] = 0;

    gpuBuffer.fromGpu();

    for (int i = 0; i < n; i++)
    {
      if (buffer[i] != gpuBuffer[i])
        std::cout << "At " << i << ": " << gpuBuffer[i] << " (expected: " << buffer[i] << ")" << std::endl;
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
