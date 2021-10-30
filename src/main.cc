#include <vulkan/vulkan.hpp>

#include <chrono>
#include <iostream>
#include <vector>
#include <thread>

#include <elasticize/window/window_manager.h>
#include <elasticize/window/window.h>
#include <elasticize/gpu/engine.h>

int main()
{
  try
  {
    elastic::window::WindowManager windowManager;
    elastic::window::Window window(1600, 900, "Elasticize");

    elastic::gpu::Engine engine;
    engine.attachWindow(window);

    while (!window.shouldClose())
    {
      windowManager.pollEvents();
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
