#include <ranges>
#include <span>

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
auto prod(T x){
  return prod(std::span<const std::ranges::range_value_t<T>>(x));
}