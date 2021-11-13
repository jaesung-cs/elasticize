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
#include <elasticize/gpu/graphics_shader.h>
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

    elastic::window::Window window(1600, 900, "Animation");
    engine.attachWindow(window);

    elastic::gpu::DescriptorSetLayout descriptorSetLayout(engine, 1);

    // Render buffer
    elastic::gpu::Buffer<float> buffer(engine, {
      0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
      1.f, 0.f, 0.f, 0.f, 1.f, 0.f,
      0.f, 1.f, 0.f, 0.f, 0.f, 1.f,
      });

    // Graphics shader
    const std::string shaderDirpath = "C:\\workspace\\elasticize\\src\\elasticize\\shader";
    elastic::gpu::GraphicsShader::Options graphicsShaderOptions;
    graphicsShaderOptions.pDescriptorSetLayout = &descriptorSetLayout;
    graphicsShaderOptions.shaders = {
      {vk::ShaderStageFlagBits::eVertex, shaderDirpath + "\\graphics\\color.vert.spv"},
      {vk::ShaderStageFlagBits::eFragment, shaderDirpath + "\\graphics\\color.frag.spv"},
    };
    graphicsShaderOptions.bindings = {
      {0, sizeof(float) * 6},
    };
    graphicsShaderOptions.attributes = {
      {0, 0, vk::Format::eR32G32B32Sfloat, 0},
      {0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3},
    };
    elastic::gpu::GraphicsShader graphicShader(engine, graphicsShaderOptions);

    window.setKeyboardCallback([&window](int key, int action)
      {
        if (key == '`' && action == 1)
          window.close();
      });

    while (!window.shouldClose())
    {
      windowManager.pollEvents();

      elastic::gpu::Execution execution(engine);
      execution.run();

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
