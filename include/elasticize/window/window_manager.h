#ifndef ELASTICIZE_WINDOW_WINDOW_MANAGER_H_
#define ELASTICIZE_WINDOW_WINDOW_MANAGER_H_

namespace elastic
{
namespace window
{
class WindowManager
{
public:
  WindowManager();
  ~WindowManager();

  void pollEvents();

private:
};
}
}

#endif // ELASTICIZE_WINDOW_WINDOW_MANAGER_H_
