#include <elasticize/window/window.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace elastic
{
namespace window
{
namespace
{
void key(GLFWwindow* windowHandle, int key, int scancode, int action, int mods)
{
  auto window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(windowHandle));
  window->keyboard(key, action);
}
}

Window::Window(uint32_t width, uint32_t height, const std::string& title)
{
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(size_[0], size_[1], title.c_str(), nullptr, nullptr);

  glfwSetWindowPos(window_, pos_[0], pos_[1]);

  glfwSetWindowSizeLimits(window_, 100, 100, GLFW_DONT_CARE, GLFW_DONT_CARE);

  // Callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetKeyCallback(window_, key);
}

Window::~Window()
{
  glfwDestroyWindow(window_);
}

bool Window::shouldClose() const
{
  return glfwWindowShouldClose(window_);
}

vk::SurfaceKHR Window::createVulkanSurface(vk::Instance instance) const
{
  VkSurfaceKHR surface;
  glfwCreateWindowSurface(instance, window_, nullptr, &surface);
  return surface;
}

void Window::close()
{
  glfwSetWindowShouldClose(window_, true);
}

void Window::setKeyboardCallback(KeyboardFn keyboardCallback)
{
  keyboardCallback_ = keyboardCallback;
}

void Window::keyboard(int key, int action)
{
  if (keyboardCallback_)
    keyboardCallback_(key, action);
}
}
}
