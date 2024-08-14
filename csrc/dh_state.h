#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "base_state.h"
#include "log.h"

inline std::string dh_xvec_str(const uint8_t *x, const char c) {
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

template <bool abrupt> struct DhState : public BaseState<abrupt> {
  uint8_t winner() const {
    const auto &x = this->x;
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

  std::string to_string() const {
    const auto &x = this->x;
    std::string out;

    if (winner() == 0xff) {
      out +=
          "** It is Player " + std::to_string(this->player() + 1) + "'s turn\n";
    } else {
      out +=
          "** GAME OVER -- Player " + std::to_string(winner() + 1) + " wins\n";
    }
    out += "** Player 1's board:\n";
    out += dh_xvec_str(x[0], 'X');
    out += "\n** Player 2's board:\n";
    out += dh_xvec_str(x[1], 'O');
    return out;
  }
};
