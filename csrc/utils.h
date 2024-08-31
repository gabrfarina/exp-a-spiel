#pragma once

#include <array>
#include <cstdint>
#include <ranges>
#include <span>

#include "log.h"

using Real = double;
using RealBuf = std::span<Real>;
using ConstRealBuf = std::span<const Real>;
template <typename T> using PerPlayer = std::array<T, 2>;
template <typename T> PerPlayer<T> make_per_player(const T &t, const T &u) {
  return {t, u};
}

template <typename T> T prod(std::span<const T> x) {
  T s = 1;
  for (const auto &v : x) {
    s *= v;
  }
  return s;
}

template <std::ranges::contiguous_range T>
  requires std::ranges::sized_range<T>
auto prod(const T &x) {
  return prod(std::span<const std::ranges::range_value_t<T>>(x));
}

inline bool is_valid(uint32_t mask, uint32_t action) {
#ifdef DEBUG
  CHECK(action < 9, "Invalid action %d", action);
#endif
  return mask & (1 << action);
}

template <typename T> auto sum(std::span<const T> x) {
  T s = 0;
  for (const auto &v : x) {
    s += v;
  }
  return s;
}

template <std::ranges::contiguous_range T>
  requires std::ranges::sized_range<T>
auto sum(const T &x) {
  return sum(std::span<const std::ranges::range_value_t<T>>(x));
}