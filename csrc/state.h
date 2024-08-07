#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "log.h"

inline std::string xvec_str(const uint8_t *x, const char c) {
  std::string lines[] = {
      "                _____",                    //
      "               /     \\",                  //
      "         _____/   6   \\_____",            //
      "        /     \\       /     \\",          //
      "  _____/   3   \\_____/   7   \\_____",    //
      " /     \\       /     \\       /     \\",  //
      "/   0   \\_____/   4   \\_____/   8   \\", //
      "\\       /     \\       /     \\       /", //
      " \\_____/   1   \\_____/   5   \\_____/",  //
      "       \\       /     \\       /",         //
      "        \\_____/   2   \\_____/",          //
      "              \\       /",                 //
      "               \\_____/"                   //
  };

  for (int i = 0; i < 9; ++i) {
    if (!x[i])
      continue;

    const uint8_t t = x[i] >> 1;
    const uint32_t sub_row = 7 + ((i % 3) - (i / 3)) * 2;
    const uint32_t sub_col = 4 + ((i % 3) + (i / 3)) * 7;
    lines[sub_row][sub_col - 1] = 't';
    lines[sub_row][sub_col] = '=';
    lines[sub_row][sub_col + 1] = '0' + t;

    if (x[i] & 1) {
      lines[sub_row - 2][sub_col - 2] = c;
      lines[sub_row - 2][sub_col - 1] = c;
      lines[sub_row - 2][sub_col] = c;
      lines[sub_row - 2][sub_col + 1] = c;
      lines[sub_row - 2][sub_col + 2] = c;
      lines[sub_row - 1][sub_col - 3] = c;
      lines[sub_row - 1][sub_col + 3] = c;
      lines[sub_row][sub_col - 3] = c;
      lines[sub_row][sub_col + 3] = c;
      lines[sub_row + 1][sub_col - 2] = c;
      lines[sub_row + 1][sub_col - 1] = c;
      lines[sub_row + 1][sub_col] = c;
      lines[sub_row + 1][sub_col + 1] = c;
      lines[sub_row + 1][sub_col + 2] = c;
    }
  }

  std::string repr;
  for (int i = 0; i < 13; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

struct DhState {
  uint8_t x[2][9];
  uint8_t p; // player
  uint8_t t[2];

  static DhState root() {
    return DhState{
        .x = {{0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}},
        .p = 0,
        .t = {0, 0},
    };
  }

  uint8_t player() const { return p; }

  void next(const uint8_t i) {
    ++t[p];
    x[p][i] = t[p] << 1;
    if (x[p ^ 1][i] == 0) {
      x[p][i] |= 1;
      p ^= 1;
    }
  }

  void next_abrupt(const uint8_t i) {
    ++t[p];
    x[p][i] = t[p] << 1;
    if (x[p ^ 1][i] == 0) {
      x[p][i] |= 1;
    }
    // Since this is the abrupt variant, the turn
    // always passes to the opponent.
    p ^= 1;
  }

  uint8_t winner() const {
    uint8_t a, b, c;
    a = x[0][1] & (x[0][0] | x[0][3]);
    b = x[0][4] & (x[0][3] | x[0][6]);
    c = x[0][7] & x[0][6];
    b |= (x[0][4] & (a | c));
    a |= x[0][1] & b;
    c |= x[0][7] & b;
    a = x[0][2] & (a | b);
    b = x[0][5] & (b | c);
    c = x[0][8] & c;
    if ((a | b | c) & 1)
      return 0;

    a = x[1][3] & (x[1][0] | x[1][1]);
    b = x[1][4] & (x[1][1] | x[1][2]);
    c = x[1][5] & x[1][2];
    b |= (x[1][4] & (a | c));
    a |= x[1][3] & b;
    c |= x[1][5] & b;
    a = x[1][6] & (a | b);
    b = x[1][7] & (b | c);
    c = x[1][8] & c;
    if ((a | b | c) & 1)
      return 1;

    return 0xff; // No winner
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

  std::string debug_string() const {
    std::string out;
    if (winner() == 0xff) {
      out += "** It is Player " + std::to_string(player() + 1) + "'s turn\n";
    } else {
      out +=
          "** GAME OVER -- Player " + std::to_string(winner() + 1) + " wins\n";
    }
    out += "** Player 1's board:\n";
    out += xvec_str(x[0], 'X');
    out += "\n** Player 2's board:\n";
    out += xvec_str(x[1], 'O');
    return out;
  }
};

inline uint8_t num_actions(uint64_t infoset) {
  uint8_t actions = 0;
  for (; infoset; ++actions, infoset >>= 5)
    ;
  return actions;
}

inline uint64_t parent_infoset(const uint64_t infoset) { return infoset >> 5; }
inline uint8_t parent_action(const uint64_t infoset) {
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