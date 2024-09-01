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

constexpr Real SMALL = 1e-20;
// Should trap Real being defined as a narrow type
static_assert(SMALL > std::numeric_limits<Real>::min());

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

inline void relu_normalize(RealBuf buf, const uint32_t mask) {
  Real s = 0;
  for (uint32_t i = 0; i < buf.size(); ++i) {
    auto const x = std::max<Real>(buf[i], 0) * is_valid(mask, i);
    s += x;
    buf[i] = x;
  }
  if (s < SMALL) {
    s = 0;
    for (uint32_t i = 0; i < buf.size(); ++i) {
      buf[i] = is_valid(mask, i);
      s += buf[i];
    }
  }
  for (uint32_t i = 0; i < buf.size(); ++i) {
    buf[i] /= s;
  }
}

inline Real dot(ConstRealBuf a, ConstRealBuf b) {
  Real s = 0;
  CHECK(a.size() == b.size(), "Vector size mismatch (expected %ld, found %ld)",
        a.size(), b.size());
  for (uint32_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}