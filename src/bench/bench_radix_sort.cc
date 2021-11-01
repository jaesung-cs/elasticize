#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/buffer.h>
#include <elasticize/utils/timer.h>

#include <iostream>
#include <random>
#include <string>
#include <iomanip>

int main()
{
  try
  {
    elastic::gpu::Engine::Options options;
    options.headless = true;
    options.validationLayer = true;
    elastic::gpu::Engine engine(options);

    std::cout << "Engine started!" << std::endl;

    const std::string shaderDirpath = "C:\\workspace\\elasticize\\src\\elasticize\\shader";
    engine.addComputeShader(shaderDirpath + "\\count.comp.spv");

    constexpr int n = 10000000;
    constexpr int keyBits = 30; // for 10-bit each component morton code
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_SIZE = 256;

    std::mt19937 gen(1234);
    std::uniform_int_distribution<uint32_t> distribution(0, (1 << keyBits) - 1);

    std::vector<uint32_t> buffer(n);
    for (int i = 0; i < n; i++)
      buffer[i] = distribution(gen);

    elastic::gpu::Buffer<uint32_t> arrayBuffer(engine, n);
    for (int i = 0; i < n; i++)
      arrayBuffer[i] = buffer[i];

    elastic::gpu::Buffer<uint32_t> counterBuffer(engine, (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE); // 1D index of [workgroupID][key]
    for (int i = 0; i < counterBuffer.size(); i++)
      counterBuffer[i] = 0;

    engine.addDescriptorSet(arrayBuffer, counterBuffer);

    // Move input from CPU to GPU
    std::cout << "To GPU" << std::endl;
    elastic::utils::Timer toGpuTimer;
    arrayBuffer.toGpu();
    std::cout << "Elapsed: " << toGpuTimer.elapsed() << std::endl;

    // Count job in GPU
    std::cout << "Count in GPU" << std::endl;
    elastic::utils::Timer gpuCountTimer;
    constexpr int bitOffset = 0;
    engine.runComputeShader(n, bitOffset);
    std::cout << "Elapsed: " << gpuCountTimer.elapsed() << std::endl;

    // Count job in CPU
    std::cout << "Count in CPU" << std::endl;
    elastic::utils::Timer cpuCountTimer;
    std::vector<uint32_t> cpuCounterBuffer(n);
    const auto groupSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int blockIndex = 0; blockIndex < groupSize; blockIndex++)
    {
      std::vector<uint32_t> localCounter(RADIX_SIZE, 0);

      // Local counter
      for (int i = 0; i < BLOCK_SIZE; i++)
      {
        const auto index = blockIndex * BLOCK_SIZE + i;
        if (index < n)
        {
          const auto item = arrayBuffer[index];
          const auto key = (static_cast<uint32_t>(item) >> bitOffset) & (RADIX_SIZE - 1);
          localCounter[key]++;
        }
      }

      // Prefix sum for this block
      uint32_t sum = 0;
      for (int i = 0; i < RADIX_SIZE; i++)
      {
        const auto count = localCounter[i];
        localCounter[i] = sum;
        sum += count;
      }

      // Move to cpu counter buffer
      for (int i = 0; i < RADIX_SIZE; i++)
      {
        const auto index = blockIndex * BLOCK_SIZE + i;
        cpuCounterBuffer[index] = localCounter[i];
      }
    }
    std::cout << "Elapsed: " << cpuCountTimer.elapsed() << std::endl;

    // Move result from GPU to CPU
    std::cout << "From GPU" << std::endl;
    elastic::utils::Timer fromGpuTimer;
    counterBuffer.fromGpu();
    std::cout << "Elapsed: " << fromGpuTimer.elapsed() << std::endl;

    // Validate
    std::cout << "Validating" << std::endl;
    for (int i = 0; i < counterBuffer.size(); i++)
    {
      if (cpuCounterBuffer[i] != counterBuffer[i])
        std::cout << "Mismatch at " << std::setw(8) << i << ": value " << counterBuffer[i] << " (expected: " << cpuCounterBuffer[i] << ")" << std::endl;
    }
    std::cout << "Validation done" << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
