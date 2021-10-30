#ifndef ELASTICIZE_WINDOW_WINDOW_H_
#define ELASTICIZE_WINDOW_WINDOW_H_

#include <string>

#include <glm/glm.hpp>

struct GLFWwindow;

namespace elastic
{
namespace window
{
class Window
{
public:
  Window(uint32_t width, uint32_t height, const std::string& title = "");
  ~Window();

  bool shouldClose() const;

private:
  glm::uvec2 pos_{ 100, 100 };
  glm::uvec2 size_{ 1600, 900 };
  GLFWwindow* window_ = nullptr;
};
}
}

#endif // ELASTICIZE_WINDOW_WINDOW_H_
