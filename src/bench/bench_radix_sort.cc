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

    elastic::gpu::Buffer<uint32_t> buffer(engine, n);
    for (int i = 0; i < n; i++)
      buffer[i] = distribution(gen);

    buffer.toGpu();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
