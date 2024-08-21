#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "log.h"

const uint8_t TIE = 0xee;

template <bool abrupt> struct BaseState {
  uint8_t x[2][9];
  uint8_t p; // player
  uint8_t t[2];

  BaseState()
      : x{{0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}}, p{0},
        t{0, 0} {}

  uint8_t player() const { return p; }

  void next(const uint8_t i) {
    ++t[p];
    x[p][i] = t[p] << 1;
    if (x[p ^ 1][i] == 0) {
      x[p][i] |= 1;
      if constexpr (!abrupt)
        p ^= 1;
    }
    if constexpr (abrupt) {
      // In the abrupt variant, the turn
      // always passes to the opponent.
      p ^= 1;
    }
  }

  uint32_t available_actions() const {
    uint32_t actions = 0;
    for (int i = 0; i < 9; ++i) {
      if (!x[p][i]) {
        actions |= (1 << i);
      }
    }
    return actions;
  }

  uint64_t get_infoset() const {
    uint64_t info = 0;
    uint8_t t_ = t[p];
    for (int i = 0; i < 9; ++i) {
      const uint8_t to = x[p][i];
      const uint8_t td = t_ - (x[p][i] >> 1);
      assert(td <= 9);
      info |= uint64_t(((i + 1) << 1) + (to & 1)) << (5 * td);
    }
    info &= (uint64_t(1) << (5 * t_)) - 1;
    return info;
  }
};

inline uint8_t num_actions(uint64_t infoset) {
  uint8_t actions = 0;
  for (; infoset; ++actions, infoset >>= 5)
    ;
  return actions;
}

inline uint64_t parent_infoset(const uint64_t infoset) {
  assert(infoset);
  return infoset >> 5;
}
inline uint8_t parent_action(const uint64_t infoset) {
  assert(infoset);
  return ((infoset >> 1) & 0b1111) - 1;
}

inline std::array<uint8_t, 9> infoset_xvec(uint64_t infoset) {
  const uint8_t na = num_actions(infoset);
  std::array<uint8_t, 9> x = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = na; infoset; --i, infoset >>= 5) {
    const uint8_t co = infoset & 0b11111;
    assert(co < 18 && i >= 1);
    x[(co >> 1) - 1] = (i << 1) + (co & 1);
  }
  return x;
}