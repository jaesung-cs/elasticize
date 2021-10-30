#include <elasticize/window/window_manager.h>

#include <GLFW/glfw3.h>

namespace elastic
{
namespace window
{
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
