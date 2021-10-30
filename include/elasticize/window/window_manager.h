#ifndef ELASTICIZE_WINDOW_WINDOW_MANAGER_H_
#define ELASTICIZE_WINDOW_WINDOW_MANAGER_H_

#include <vector>

namespace elastic
{
namespace window
{
class WindowManager
{
public:
  static std::vector<const char*> requiredInstanceExtensions();

public:
  WindowManager();
  ~WindowManager();

  void pollEvents();

private:
};
}
}

#endif // ELASTICIZE_WINDOW_WINDOW_MANAGER_H_
