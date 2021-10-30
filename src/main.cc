#include <vulkan/vulkan.hpp>

#include <chrono>
#include <iostream>
#include <vector>
#include <thread>

#include <elasticize/window/window_manager.h>
#include <elasticize/window/window.h>

int main()
{
  {
    elastic::window::WindowManager windowManager;
    elastic::window::Window window(1600, 900, "Elasticize");

    while (!window.shouldClose())
    {
      windowManager.pollEvents();
    }
  }

  std::vector<const char*> layers;
#if defined(VULKAN_VALIDATION)
  layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

  const auto appInfo = vk::ApplicationInfo()
    .setPApplicationName("Vulkan C++ Program Template")
    .setApplicationVersion(1)
    .setPEngineName("LunarG SDK")
    .setEngineVersion(1)
    .setApiVersion(VK_API_VERSION_1_0);

  const auto instanceInfo = vk::InstanceCreateInfo()
    .setFlags(vk::InstanceCreateFlags())
    .setPApplicationInfo(&appInfo)
    .setEnabledExtensionCount(0)
    .setPpEnabledExtensionNames(NULL)
    .setEnabledLayerCount(static_cast<uint32_t>(layers.size()))
    .setPpEnabledLayerNames(layers.data());

  vk::Instance instance;
  try
  {
    instance = vk::createInstance(instanceInfo);
  }
  catch (const std::exception& e)
  {
    std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
    return 1;
  }
  instance.destroy();

  return 0;
}
