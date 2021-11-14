#ifndef ELASTICIZE_WINDOW_WINDOW_H_
#define ELASTICIZE_WINDOW_WINDOW_H_

#include <string>

#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

struct GLFWwindow;

namespace elastic
{
namespace window
{
class Window
{
public:
  using KeyboardFn = std::function<void(int, int)>;

public:
  Window(uint32_t width, uint32_t height, const std::string& title = "");
  ~Window();

  bool shouldClose() const;
  auto width() const { return size_[0]; }
  auto height() const { return size_[1]; }
  void close();

  void setKeyboardCallback(KeyboardFn keyboardCallback);

  vk::SurfaceKHR createVulkanSurface(vk::Instance instance) const;

  void keyboard(int key, int action);

private:
  glm::uvec2 pos_{ 100, 100 };
  glm::uvec2 size_{ 1600, 900 };
  GLFWwindow* window_ = nullptr;

  KeyboardFn keyboardCallback_;
};
}
}

#endif // ELASTICIZE_WINDOW_WINDOW_H_
