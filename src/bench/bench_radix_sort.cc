#include <iostream>
#include <random>
#include <string>
#include <iomanip>
#include <execution>

#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/buffer.h>
#include <elasticize/gpu/descriptor_set_layout.h>
#include <elasticize/gpu/descriptor_set.h>
#include <elasticize/gpu/compute_shader.h>
#include <elasticize/utils/timer.h>

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
    engine.addComputeShader(shaderDirpath + "\\scan_forward.comp.spv");
    engine.addComputeShader(shaderDirpath + "\\scan_backward.comp.spv");
    engine.addComputeShader(shaderDirpath + "\\distribute.comp.spv");

    constexpr int n = 1000000;
    constexpr int keyBits = 30; // for 10-bit each component morton code
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_SIZE = 256;
    constexpr int RADIX_BITS = 8;

    std::mt19937 gen(1234);
    std::uniform_int_distribution<uint32_t> distribution(0, (1 << keyBits) - 1);

    struct KeyValue
    {
      uint32_t key;
      uint32_t value;

      bool operator != (const KeyValue& rhs) const
      {
        return key != rhs.key || value != rhs.value;
      }
    };

    std::vector<KeyValue> buffer(n);
    for (uint32_t i = 0; i < n; i++)
      buffer[i] = { distribution(gen), i };

    elastic::gpu::Buffer<KeyValue> arrayBuffer(engine, n);
    elastic::gpu::Buffer<KeyValue> outBuffer(engine, n);
    for (int i = 0; i < n; i++)
      arrayBuffer[i] = buffer[i];

    const auto alignedSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t counterSize = 0;
    uint32_t simdSize = alignedSize;
    do
    {
      counterSize += simdSize;
      simdSize = (simdSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    } while (simdSize > 1);

    elastic::gpu::Buffer<uint32_t> counterBuffer(engine, counterSize); // 1D index of [workgroupID][key]
    for (int i = 0; i < counterBuffer.size(); i++)
      counterBuffer[i] = 0;

    engine.addDescriptorSet(arrayBuffer, counterBuffer, outBuffer);

    elastic::gpu::DescriptorSetLayout descriptorSetLayout(engine, 3);
    elastic::gpu::DescriptorSet descriptorSet(engine, descriptorSetLayout, {
      arrayBuffer,
      counterBuffer,
      outBuffer,
      });

    elastic::gpu::ComputeShader countShader(engine, shaderDirpath + "\\count.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader scanForwardShader(engine, shaderDirpath + "\\scan_forward.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader scanBackwardShader(engine, shaderDirpath + "\\scan_backward.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader distributeShader(engine, shaderDirpath + "\\distribute.comp.spv", descriptorSetLayout, {});

    // Move input from CPU to GPU
    std::cout << "To GPU" << std::endl;
    elastic::utils::Timer toGpuTimer;
    arrayBuffer.toGpu();
    std::cout << "Elapsed: " << toGpuTimer.elapsed() << std::endl;

    for (int bitOffset = 0; bitOffset < keyBits; bitOffset += RADIX_BITS)
    {
      // Count job in GPU
      std::cout << "Count in GPU" << std::endl;
      elastic::utils::Timer gpuCountTimer;
      engine.runComputeShader(0, n, bitOffset);
      std::cout << "Elapsed: " << gpuCountTimer.elapsed() << std::endl;

      // Scan forward job in GPU
      std::cout << "Scan forward in GPU" << std::endl;
      elastic::utils::Timer gpuScanTimer;
      uint32_t scanOffset = 0;
      simdSize = alignedSize;

      struct Phase
      {
        uint32_t simdSize;
        uint32_t scanOffset;
      };
      std::vector<Phase> phases;
      do
      {
        std::cout << "array size, scan offset: " << simdSize << ", " << scanOffset << std::endl;
        phases.push_back(Phase{ simdSize, scanOffset });
        engine.runComputeShader(1, simdSize, bitOffset, scanOffset);
        scanOffset += simdSize;
        simdSize = (simdSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
      } while (simdSize > 1);
      std::cout << "Elapsed: " << gpuScanTimer.elapsed() << std::endl;

      // Scan backward job in GPU
      std::cout << "Scan backward in GPU" << std::endl;
      elastic::utils::Timer gpuScanBackwardTimer;
      for (int i = phases.size() - 1; i >= 0; i--)
      {
        const auto& phase = phases[i];
        engine.runComputeShader(2, phase.simdSize, bitOffset, phase.scanOffset);
      }
      std::cout << "Elapsed: " << gpuScanBackwardTimer.elapsed() << std::endl;

      // Now prefix sum of counter[key][block_index], meaning start offset of each group
      std::cout << "Distribute in GPU" << std::endl;
      elastic::utils::Timer gpuDistributeTimer;
      engine.runComputeShader(3, n, bitOffset);
      std::cout << "Elapsed: " << gpuDistributeTimer.elapsed() << std::endl;

      // Copy to input buffer
      engine.copyBuffer(outBuffer, arrayBuffer);
    }
    
    // Move result from GPU to CPU
    std::cout << "From GPU" << std::endl;
    elastic::utils::Timer fromGpuTimer;
    arrayBuffer.fromGpu();
    std::cout << "Elapsed: " << fromGpuTimer.elapsed() << std::endl;

    // CPU radix sort
    std::cout << "CPU radix sort" << std::endl;
    elastic::utils::Timer cpuRadixSortTimer;
    std::sort(std::execution::par,
      buffer.begin(), buffer.end(), [RADIX_SIZE](const KeyValue& lhs, const KeyValue& rhs)
      {
        return lhs.key < rhs.key;
      });
    std::cout << "Elapsed: " << cpuRadixSortTimer.elapsed() << std::endl;

    // Validate
    std::cout << "Validating count" << std::endl;
    for (int i = 0; i < n; i++)
    {
      if (buffer[i].key != arrayBuffer[i].key)
        std::cout << "Buffer mismatch at " << std::setw(8) << i << ": value " << std::hex << std::setw(8) << arrayBuffer[i].key << " (expected: " << std::setw(8) << buffer[i].key << std::dec << ")" << std::endl;
    }
    std::cout << "Validation done" << std::endl;

    std::cout << "First two block:" << std::endl;
    for (int i = 0; i < BLOCK_SIZE * 2; i++)
      std::cout << std::setw(4) << i << ' ' << std::setw(4) << arrayBuffer[i].key << ' ' << std::setw(4) << arrayBuffer[i].value << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
