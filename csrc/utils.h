#include <ranges>
#include <span>
#include "log.h"
template<typename T>
T prod(std::span<const T> x){
  T s = 1;
  for (const auto& v : x) {
    s *= v;
  }
  return s;
}

template <std::ranges::contiguous_range T>
    requires std::ranges::sized_range<T>
auto prod(const T& x){
  return prod(std::span<const std::ranges::range_value_t<T>>(x));
}

inline bool is_valid(uint32_t mask, uint32_t action){
  #ifdef DEBUG
  CHECK(action < 9, "Invalid action %d", action);
  #endif
  return mask & (1 << action);
}

template<typename T>
auto sum(std::span<const T> x){
  T s = 0;
  for (const auto& v : x) {
    s += v;
  }
  return s;
}

template <std::ranges::contiguous_range T>
    requires std::ranges::sized_range<T>
auto sum(const T& x){
  return sum(std::span<const std::ranges::range_value_t<T>>(x));
}