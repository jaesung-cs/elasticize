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

    elastic::window::Window window(1600, 900, "Animation");
    engine.attachWindow(window);

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
