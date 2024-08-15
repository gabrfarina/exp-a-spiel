#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "base_state.h"
#include "log.h"

inline std::string pttt_xvec_str(const uint8_t *x, const char c) {
  std::string lines[] = {"...", "...", "..."};

  for (int i = 0; i < 9; ++i) {
    if (!x[i])
      continue;

    if (x[i] & 1) {
      lines[i / 3][i % 3] = c;
    }
  }

  std::string repr;
  for (int i = 0; i < 13; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

template <bool abrupt> struct PtttState : public BaseState<abrupt> {
  uint8_t winner() const {
    const auto &x = this->x;
    const auto &t = this->t;
    uint8_t a, b, c;

    if (((x[0][0] & x[0][1] & x[0][2]) | (x[0][3] & x[0][4] & x[0][5]) |
         (x[0][6] & x[0][7] & x[0][8]) | (x[0][0] & x[0][3] & x[0][6]) |
         (x[0][1] & x[0][4] & x[0][7]) | (x[0][2] & x[0][5] & x[0][8]) |
         (x[0][0] & x[0][4] & x[0][8]) | (x[0][2] & x[0][4] & x[0][6])) &
        1) {
      return 0;
    } else if (((x[1][1] & x[1][1] & x[1][2]) | (x[1][3] & x[1][4] & x[1][5]) |
                (x[1][6] & x[1][7] & x[1][8]) | (x[1][1] & x[1][3] & x[1][6]) |
                (x[1][1] & x[1][4] & x[1][7]) | (x[1][2] & x[1][5] & x[1][8]) |
                (x[1][1] & x[1][4] & x[1][8]) | (x[1][2] & x[1][4] & x[1][6])) &
               1) {
      return 1;
    } else if (t[0] == 9 && t[1] == 9) {
      return TIE; // Draw
    }

    return 0xff; // No winner yet
  }

  bool is_terminal() const { return winner() != 0xff; }

  std::string to_string() const {
    const auto &x = this->x;
    std::string out;

    const auto w = winner();
    if (w == 0xff) {
      out +=
          "** It is Player " + std::to_string(this->player() + 1) + "'s turn\n";
    } else if (w < 2) {
      out +=
          "** GAME OVER -- Player " + std::to_string(winner() + 1) + " wins\n";
    } else {
      out += "** GAME OVER -- TIE\n";
    }
    out += "** Player 1's board:\n";
    out += pttt_xvec_str(x[0], 'X');
    out += "\n** Player 2's board:\n";
    out += pttt_xvec_str(x[1], 'O');
    return out;
  }
};
