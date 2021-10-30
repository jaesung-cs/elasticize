#include <elasticize/window/window_manager.h>

#include <GLFW/glfw3.h>

namespace elastic
{
namespace window
{
std::vector<const char*> WindowManager::requiredInstanceExtensions()
{
  uint32_t count;
  const char** instanceExtensions = glfwGetRequiredInstanceExtensions(&count);

  std::vector<const char*> result;
  for (uint32_t i = 0; i < count; i++)
    result.push_back(instanceExtensions[i]);
  return result;
}

WindowManager::WindowManager()
{
  glfwInit();
}

WindowManager::~WindowManager()
{
  glfwTerminate();
}

void WindowManager::pollEvents()
{
  glfwPollEvents();
}
}
}
