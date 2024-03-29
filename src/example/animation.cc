#include <iostream>
#include <random>
#include <string>
#include <iomanip>
#include <execution>

#include <elasticize/window/window_manager.h>
#include <elasticize/window/window.h>
#include <elasticize/gpu/engine.h>
#include <elasticize/gpu/swapchain.h>
#include <elasticize/gpu/buffer.h>
#include <elasticize/gpu/image.h>
#include <elasticize/gpu/descriptor_set_layout.h>
#include <elasticize/gpu/descriptor_set.h>
#include <elasticize/gpu/compute_shader.h>
#include <elasticize/gpu/graphics_shader.h>
#include <elasticize/gpu/framebuffer.h>
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

    // Descriptor
    elastic::gpu::DescriptorSetLayout descriptorSetLayout(engine, 1);
    elastic::gpu::DescriptorSet descriptorSet(engine, descriptorSetLayout, {});

    // Render buffer
    elastic::gpu::Buffer<float> vertexBuffer(engine, {
      0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
      1.f, 0.f, 0.f, 0.f, 1.f, 0.f,
      0.f, 1.f, 0.f, 0.f, 0.f, 1.f,
      });

    elastic::gpu::Buffer<uint32_t> indexBuffer(engine, {
      0, 1, 2,
      });

    elastic::gpu::Execution(engine)
      .toGpu(vertexBuffer)
      .toGpu(indexBuffer)
      .end()
      .run();

    // Swapchain
    constexpr uint32_t width = 1600;
    constexpr uint32_t height = 900;
    elastic::window::Window window(width, height, "Animation");
    elastic::gpu::Swapchain swapchain(engine, window);
    const auto& swapchainInfo = swapchain.info();

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
    graphicsShaderOptions.imageFormat = swapchainInfo.imageFormat;
    elastic::gpu::GraphicsShader graphicsShader(engine, graphicsShaderOptions);

    // Image views
    constexpr vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e4;

    elastic::gpu::Image::Options imageOptions;
    imageOptions.width = swapchainInfo.imageExtent.width;
    imageOptions.height = swapchainInfo.imageExtent.height;
    imageOptions.samples = samples;
    imageOptions.format = swapchainInfo.imageFormat;
    imageOptions.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment;
    elastic::gpu::Image transientColorImage(engine, imageOptions);

    imageOptions.format = vk::Format::eD24UnormS8Uint;
    imageOptions.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment;
    elastic::gpu::Image transientDepthImage(engine, imageOptions);

    // Framebuffer
    std::vector<elastic::gpu::Framebuffer> framebuffers;
    for (auto swapchainImage : swapchain.images())
    {
      framebuffers.push_back(elastic::gpu::Framebuffer(engine, width, height, graphicsShader,
        {
          transientColorImage,
          transientDepthImage,
          swapchainImage,
        }));
    }
    
    // Rendering command
    std::vector<elastic::gpu::Execution> drawCommands;
    for (uint32_t i = 0; i < swapchain.imageCount(); i++)
    {
      drawCommands.emplace_back(engine);
      drawCommands[i]
        .draw(graphicsShader, descriptorSet, framebuffers[i], vertexBuffer, indexBuffer)
        .end();
    }

    window.setKeyboardCallback([&window](int key, int action)
      {
        if (key == '`' && action == 1)
          window.close();
      });

    // TODO: image available semaphore
    auto device = engine.device();
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> renderFinishedFences;
    for (int i = 0; i < swapchain.imageCount(); i++)
    {
      imageAvailableSemaphores.push_back(device.createSemaphore({}));
      renderFinishedSemaphores.push_back(device.createSemaphore({}));
      renderFinishedFences.push_back(device.createFence({ vk::FenceCreateFlagBits::eSignaled }));
    }

    uint64_t frame = 0;
    while (!window.shouldClose())
    {
      windowManager.pollEvents();

      uint32_t frameIndex = frame % swapchain.imageCount();

      device.waitForFences(renderFinishedFences[frameIndex], true, UINT64_MAX);
      device.resetFences(renderFinishedFences[frameIndex]);

      auto imageIndex = swapchain.acquireNextImage(imageAvailableSemaphores[frameIndex]);
      std::cout << imageIndex << std::endl;

      drawCommands[imageIndex].present(imageAvailableSemaphores[frameIndex], renderFinishedSemaphores[imageIndex], renderFinishedFences[frameIndex], swapchain, imageIndex);

      using namespace std::chrono_literals;
      std::this_thread::sleep_for(0.01s);
    }

    for (int i = 0; i < swapchain.imageCount(); i++)
    {
      device.destroySemaphore(imageAvailableSemaphores[i]);
      device.destroySemaphore(renderFinishedSemaphores[i]);
      device.destroyFence(renderFinishedFences[i]);
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
