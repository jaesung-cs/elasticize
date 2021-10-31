#include <elasticize/utils/timer.h>

namespace elastic
{
namespace utils
{
Timer::Timer()
{
  timePoint_ = Clock::now();
}

Timer::~Timer() = default;

double Timer::elapsed() const
{
  return Duration(Clock::now() - timePoint_).count();
}
}
}
