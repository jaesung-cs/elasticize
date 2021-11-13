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
#include <elasticize/gpu/execution.h>
#include <elasticize/utils/timer.h>

int main()
{
  try
  {
    elastic::gpu::Engine::Options options;
    options.headless = true;
    options.validationLayer = true;
    options.memoryPoolSize = 256 * 1024 * 1024; // 256MB
    elastic::gpu::Engine engine(options);

    std::cout << "Engine started!" << std::endl;

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

    elastic::gpu::DescriptorSetLayout descriptorSetLayout(engine, 3);
    elastic::gpu::DescriptorSet descriptorSet(engine, descriptorSetLayout, {
      arrayBuffer,
      counterBuffer,
      outBuffer,
      });

    const std::string shaderDirpath = "C:\\workspace\\elasticize\\src\\elasticize\\shader";
    elastic::gpu::ComputeShader countShader(engine, shaderDirpath + "\\radix_sort\\count.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader scanForwardShader(engine, shaderDirpath + "\\radix_sort\\scan_forward.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader scanBackwardShader(engine, shaderDirpath + "\\radix_sort\\scan_backward.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader distributeShader(engine, shaderDirpath + "\\radix_sort\\distribute.comp.spv", descriptorSetLayout, {});

    // Move to GPU
    elastic::gpu::Execution(engine).toGpu(arrayBuffer).run();

    // Radix sort
    elastic::gpu::Execution execution(engine);
    struct SortInfoUbo
    {
      uint32_t array_size;
      int32_t bit_offset;
      uint32_t scan_offset;
    };
    SortInfoUbo sortInfo;

    for (int bitOffset = 0; bitOffset < keyBits; bitOffset += RADIX_BITS)
    {
      sortInfo = { n, bitOffset, 0 };
      execution
        .runComputeShader(countShader, descriptorSet, (n + BLOCK_SIZE - 1) / BLOCK_SIZE, sortInfo)
        .barrier();

      // Scan forward
      struct Phase
      {
        uint32_t simdSize;
        uint32_t scanOffset;
      };
      std::vector<Phase> phases;
      uint32_t scanOffset = 0;
      simdSize = alignedSize;
      do
      {
        phases.push_back(Phase{ simdSize, scanOffset });
        sortInfo = { simdSize, bitOffset, scanOffset };
        execution
          .runComputeShader(scanForwardShader, descriptorSet, (simdSize + BLOCK_SIZE - 1) / BLOCK_SIZE, sortInfo)
          .barrier();
        scanOffset += simdSize;
        simdSize = (simdSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
      } while (simdSize > 1);

      // Scan backward
      for (int i = phases.size() - 1; i >= 0; i--)
      {
        const auto& phase = phases[i];
        sortInfo = { phase.simdSize, bitOffset, phase.scanOffset };
        execution
          .runComputeShader(scanBackwardShader, descriptorSet, (phase.simdSize + BLOCK_SIZE - 1) / BLOCK_SIZE, sortInfo)
          .barrier();
      }

      // Now prefix sum of counter[key][block_index], meaning start offset of each group
      sortInfo = { n, bitOffset, 0 };
      execution
        .runComputeShader(distributeShader, descriptorSet, (n + BLOCK_SIZE - 1) / BLOCK_SIZE, sortInfo)
        .barrier();

      // Copy to input buffer
      execution
        .copy(outBuffer, arrayBuffer)
        .barrier();
    }

    std::cout << "GPU radix sort" << std::endl;
    elastic::utils::Timer gpuTimer;
    execution.run();
    std::cout << "Elapsed: " << gpuTimer.elapsed() << std::endl;

    // From GPU
    elastic::gpu::Execution(engine).fromGpu(arrayBuffer).run();

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
