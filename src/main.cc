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

    elastic::gpu::Engine::Options options;
    options.headless = false;
    options.validationLayer = true;

    elastic::gpu::Engine engine(options);
    engine.attachWindow(window);

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
