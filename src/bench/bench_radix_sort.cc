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
    engine.addComputeShader(shaderDirpath + "\\scan.comp.spv");

    constexpr int n = 1000000;
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

    const auto alignedSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    elastic::gpu::Buffer<uint32_t> counterBuffer(engine, alignedSize); // 1D index of [workgroupID][key]
    for (int i = 0; i < counterBuffer.size(); i++)
      counterBuffer[i] = 0;

    elastic::gpu::Buffer<uint32_t> scanBuffer(engine, alignedSize); // 1D index of [workgroupID][key]
    for (int i = 0; i < scanBuffer.size(); i++)
      scanBuffer[i] = 0;

    engine.addDescriptorSet(arrayBuffer, counterBuffer, scanBuffer);

    // Move input from CPU to GPU
    std::cout << "To GPU" << std::endl;
    elastic::utils::Timer toGpuTimer;
    arrayBuffer.toGpu();
    std::cout << "Elapsed: " << toGpuTimer.elapsed() << std::endl;

    // Count job in GPU
    std::cout << "Count in GPU" << std::endl;
    elastic::utils::Timer gpuCountTimer;
    constexpr int bitOffset = 0;
    engine.runComputeShader(0, n, bitOffset);
    std::cout << "Elapsed: " << gpuCountTimer.elapsed() << std::endl;

    // Count job in CPU
    std::cout << "Count in CPU" << std::endl;
    elastic::utils::Timer cpuCountTimer;
    std::vector<uint32_t> cpuCounterBuffer(counterBuffer.size());
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

      // Move to cpu counter buffer
      for (int i = 0; i < RADIX_SIZE; i++)
      {
        const auto index = i * groupSize + blockIndex;
        cpuCounterBuffer[index] = localCounter[i];
      }
    }
    std::cout << "Elapsed: " << cpuCountTimer.elapsed() << std::endl;

    // Scan job in GPU
    std::cout << "Scan in GPU" << std::endl;
    elastic::utils::Timer gpuScanTimer;
    engine.runComputeShader(1, alignedSize, bitOffset);
    std::cout << "Elapsed: " << gpuScanTimer.elapsed() << std::endl;

    // Scan job in CPU
    std::cout << "Scan in CPU" << std::endl;
    elastic::utils::Timer cpuScanTimer;
    std::vector<uint32_t> cpuScanBuffer(alignedSize);
    for (int blockIndex = 0; blockIndex < groupSize; blockIndex++)
    {
      // Exclusive scan
      uint32_t sum = 0;
      for (int i = 0; i < BLOCK_SIZE; i++)
      {
        const auto index = blockIndex * BLOCK_SIZE + i;
        const auto count = cpuCounterBuffer[index];
        cpuScanBuffer[index] = sum;
        sum += count;
      }
    }
    std::cout << "Elapsed: " << cpuScanTimer.elapsed() << std::endl;

    // Move result from GPU to CPU
    std::cout << "From GPU" << std::endl;
    elastic::utils::Timer fromGpuTimer;
    counterBuffer.fromGpu();
    scanBuffer.fromGpu();
    std::cout << "Elapsed: " << fromGpuTimer.elapsed() << std::endl;

    // Validate
    std::cout << "Validating count" << std::endl;
    for (int i = 0; i < counterBuffer.size(); i++)
    {
      if (cpuCounterBuffer[i] != counterBuffer[i])
        std::cout << "Mismatch at " << std::setw(8) << i << ": value " << counterBuffer[i] << " (expected: " << cpuCounterBuffer[i] << ")" << std::endl;
    }
    std::cout << "Validating scan" << std::endl;
    for (int i = 0; i < scanBuffer.size(); i++)
    {
      if (cpuScanBuffer[i] != scanBuffer[i])
        std::cout << "Mismatch at " << std::setw(8) << i << ": value " << scanBuffer[i] << " (expected: " << cpuScanBuffer[i] << ")" << std::endl;
    }
    std::cout << "Validation done" << std::endl;

    std::cout << "First two block:" << std::endl;
    for (int i = 0; i < BLOCK_SIZE * 2; i++)
      std::cout << std::setw(4) << i << ' ' << std::setw(4) << counterBuffer[i] << ' ' << std::setw(4) << scanBuffer[i] << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
