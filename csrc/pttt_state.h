#pragma once

#include <cassert>
#include <cstdint>

#include "base_state.h"

inline std::string pttt_xvec_str(const uint8_t *x, const char c) {
  std::string lines[] = {"...", "...", "..."};

  for (int i = 0; i < 9; ++i) {
    if (!x[i]) continue;

    if (x[i] & 1) {
      lines[i / 3][i % 3] = c;
    }
  }

  std::string repr;
  for (int i = 0; i < 3; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

template <bool abrupt>
struct PtttState : public BaseState<abrupt> {
  static constexpr uint32_t OPENSPIEL_INFOSTATE_SIZE = 27 + 81;

  uint8_t winner() const {
    const auto &x = this->x;

    const uint8_t num_filled_p0 = (x[0][0] & 1) + (x[0][1] & 1) +
                                  (x[0][2] & 1) + (x[0][3] & 1) +
                                  (x[0][4] & 1) + (x[0][5] & 1) +
                                  (x[0][6] & 1) + (x[0][7] & 1) + (x[0][8] & 1);
    const uint8_t num_filled_p1 = (x[1][0] & 1) + (x[1][1] & 1) +
                                  (x[1][2] & 1) + (x[1][3] & 1) +
                                  (x[1][4] & 1) + (x[1][5] & 1) +
                                  (x[1][6] & 1) + (x[1][7] & 1) + (x[1][8] & 1);

    if (((x[0][0] & x[0][1] & x[0][2]) | (x[0][3] & x[0][4] & x[0][5]) |
         (x[0][6] & x[0][7] & x[0][8]) | (x[0][0] & x[0][3] & x[0][6]) |
         (x[0][1] & x[0][4] & x[0][7]) | (x[0][2] & x[0][5] & x[0][8]) |
         (x[0][0] & x[0][4] & x[0][8]) | (x[0][2] & x[0][4] & x[0][6])) &
        1) {
      return 0;
    } else if (((x[1][0] & x[1][1] & x[1][2]) | (x[1][3] & x[1][4] & x[1][5]) |
                (x[1][6] & x[1][7] & x[1][8]) | (x[1][0] & x[1][3] & x[1][6]) |
                (x[1][1] & x[1][4] & x[1][7]) | (x[1][2] & x[1][5] & x[1][8]) |
                (x[1][0] & x[1][4] & x[1][8]) | (x[1][2] & x[1][4] & x[1][6])) &
               1) {
      return 1;
    } else if (num_filled_p0 + num_filled_p1 == 9) {
      return TIE;
    }

    return 0xff;  // No winner yet
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
      assert(w == TIE);
      out += "** GAME OVER -- TIE\n";
    }
    out += "** Player 1's board:\n";
    out += pttt_xvec_str(x[0], 'X');
    out += "\n** Player 2's board:\n";
    out += pttt_xvec_str(x[1], 'O');
    return out;
  }

  static void compute_openspiel_infostate(uint8_t player, uint64_t info,
                                          std::span<bool> buf) {
    const uint8_t n_actions = num_actions(info);
    std::fill(buf.begin(), buf.end(), 0);
    // Mark first 9 cells as empty
    std::fill(buf.begin(), buf.begin() + 9, true);
    for (uint32_t j = 0; info; info >>= 5, ++j) {
      const uint8_t cell = ((info >> 1) & 0b1111) - 1;
      const bool placed = info & 1;
      assert(cell < 9);

      buf[cell] = false;
      buf[9 + cell] = ((player + placed) % 2 == 0);  // p1 first...
      buf[18 + cell] =
          ((player + placed) % 2 == 1);  // ... then p0 (not a typo)
      // we are reading moves from latest to oldest, and we want to store moves
      // from oldest to latest
      buf[27 + 9 * (n_actions - j - 1) + cell] = true;
    }
  }
};
