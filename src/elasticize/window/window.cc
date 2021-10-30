#include <elasticize/window/window.h>

#include <GLFW/glfw3.h>

namespace elastic
{
namespace window
{
Window::Window(uint32_t width, uint32_t height, const std::string& title)
{
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(size_[0], size_[1], title.c_str(), nullptr, nullptr);

  glfwSetWindowPos(window_, pos_[0], pos_[1]);

  glfwSetWindowSizeLimits(window_, 100, 100, GLFW_DONT_CARE, GLFW_DONT_CARE);
}

Window::~Window()
{
  glfwDestroyWindow(window_);
}

bool Window::shouldClose() const
{
  return glfwWindowShouldClose(window_);
}
}
}
