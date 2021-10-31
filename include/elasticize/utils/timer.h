#ifndef ELASTICIZE_UTILS_TIMER_H_
#define ELASTICIZE_UTILS_TIMER_H_

#include <chrono>

namespace elastic
{
namespace utils
{
class Timer
{
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;
  using Duration = std::chrono::duration<double>;

public:
  Timer();
  ~Timer();

  double elapsed() const;

private:
  TimePoint timePoint_;
};
}
}

#endif // ELASTICIZE_UTILS_TIMER_H_
