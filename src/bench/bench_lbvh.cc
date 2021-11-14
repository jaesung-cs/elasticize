#include <iostream>
#include <random>
#include <string>
#include <iomanip>
#include <execution>

#include <elasticize/window/window_manager.h>
#include <elasticize/window/window.h>
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
    elastic::window::WindowManager windowManager;

    elastic::gpu::Engine::Options options;
    options.headless = false;
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

    elastic::gpu::DescriptorSetLayout descriptorSetLayout(engine, 3);

    const std::string shaderDirpath = "C:\\workspace\\elasticize\\src\\elasticize\\shader";
    elastic::gpu::ComputeShader countShader(engine, shaderDirpath + "\\radix_sort\\count.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader scanForwardShader(engine, shaderDirpath + "\\radix_sort\\scan_forward.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader scanBackwardShader(engine, shaderDirpath + "\\radix_sort\\scan_backward.comp.spv", descriptorSetLayout, {});
    elastic::gpu::ComputeShader distributeShader(engine, shaderDirpath + "\\radix_sort\\distribute.comp.spv", descriptorSetLayout, {});

    elastic::window::Window window(1600, 900, "Benchmark - LBVH");

    while (!window.shouldClose())
    {
      windowManager.pollEvents();

      using namespace std::chrono_literals;
      std::this_thread::sleep_for(0.01s);
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
