#pragma once

#include <bitset>
#include <cstdint>

#include "base_state.h"
#include "utils.h"

inline std::string dh_xvec_str(const uint8_t *x, const char c) {
  std::string lines[] = {
      "                _____",                     //
      "               /     \\",                   //
      "         _____/   2   \\_____",             //
      "        /     \\       /     \\",           //
      "  _____/   1   \\_____/   5   \\_____",     //
      " /     \\       /     \\       /     \\",   //
      "/   0   \\_____/   4   \\_____/   8   \\",  //
      "\\       /     \\       /     \\       /",  //
      " \\_____/   3   \\_____/   7   \\_____/",   //
      "       \\       /     \\       /",          //
      "        \\_____/   6   \\_____/",           //
      "              \\       /",                  //
      "               \\_____/"                    //
  };

  for (int i = 0; i < 9; ++i) {
    if (!x[i]) continue;

    const uint8_t t = x[i] >> 1;
    const uint32_t sub_row = 7 + ((i / 3) - (i % 3)) * 2;
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

template <bool abrupt>
struct DhState : public BaseState<abrupt> {
  static constexpr uint32_t num_cells = 9;
  // new openspiel uses these :upsidedown:
  static constexpr uint32_t bits_per_action = num_cells;
  static constexpr uint32_t longest_sequence = num_cells;

  // static constexpr uint32_t bits_per_action = num_cells + 1;
  // static constexpr uint32_t longest_sequence = num_cells * 2 - 1;
  // white is player 2
  // We store the sequence of actions as 9 one-hot vectors of size 9
  // For the board we store 9 one-hot vector of size 9
  enum cell_state {
    kEmpty = 0,
    kWhiteWest = -3,
    kWhiteEast = -2,
    kWhiteWin = -4,  // impossible here
    kWhite = -1,     // White and not edge connected
    kBlackNorth = 3,
    kBlackSouth = 2,
    kBlackWin = 4,  // impossible here
    kBlack = 1,     // Black and not edge connected
  };

  constexpr static int min_cell_state = -4;
  constexpr static int cell_states = 9;

  // static constexpr uint32_t OPENSPIEL_INFOSTATE_SIZE =
  //     num_cells * cell_states + longest_sequence * (1 + bits_per_action);
  static constexpr uint32_t OPENSPIEL_INFOSTATE_SIZE =
      num_cells * cell_states + longest_sequence * bits_per_action;

  uint8_t winner() const {
    const auto &x = this->x;
    uint8_t a, b, c;

    a = x[0][3] & (x[0][0] | x[0][1]);
    b = x[0][4] & (x[0][1] | x[0][2]);
    c = x[0][5] & x[0][2];
    b |= (x[0][4] & (a | c));
    a |= x[0][3] & b;
    c |= x[0][5] & b;
    a = x[0][6] & (a | b);
    b = x[0][7] & (b | c);
    c = x[0][8] & c;
    if ((a | b | c) & 1) return 0;

    a = x[1][1] & (x[1][0] | x[1][3]);
    b = x[1][4] & (x[1][3] | x[1][6]);
    c = x[1][7] & x[1][6];
    b |= (x[1][4] & (a | c));
    a |= x[1][1] & b;
    c |= x[1][7] & b;
    a = x[1][2] & (a | b);
    b = x[1][5] & (b | c);
    c = x[1][8] & c;
    if ((a | b | c) & 1) return 1;

    return 0xff;  // No winner
  }

  bool is_terminal() const { return winner() != 0xff; }

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

  static void compute_openspiel_infostate(uint8_t player, uint64_t info,
                                          std::span<bool> buf) {
    std::fill(buf.begin(), buf.end(), false);
    PerPlayer<std::bitset<9>> b{};
    int n_actions = 0, m_actions = 0;
    for (uint64_t i = info; i; i >>= 5, ++n_actions) {
      bool success = i & 1;
      uint8_t cell = ((i >> 1) & 0b1111) - 1;
      uint8_t p = success ? player : 1 - player;
      b[p][cell] = true;
    }

    for (uint64_t i = info; i; i >>= 5, ++m_actions){
      uint8_t cell = ((i >> 1) & 0b1111) - 1;
      buf[num_cells * cell_states + (n_actions - 1 - m_actions) * bits_per_action + cell] = true;
    }

    for (auto i = 0; i < 9; ++i) {
      buf[i * cell_states + kEmpty - min_cell_state] = !(b[0][i] || b[1][i]);
    }

    {
      // west is 0 3 6
      std::bitset<9> west, east;
      // west[0] = b[1][0];
      // west[3] = b[1][3];
      // west[6] = b[1][6];
      // west[1] = b[1][1] && (b[1][0] || b[1][3]);
      // west[4] = b[1][4] && (b[1][3] || b[1][6]);
      // west[7] = b[1][7] && b[1][6];
      // west[2] = b[1][2] && (west[1] || west[4]);
      // west[5] = b[1][5] && (west[4] || west[7]);
      // west[8] = b[1][8] && west[7];

      // east[2] = b[1][2];
      // east[5] = b[1][5];
      // east[8] = b[1][8];
      // east[1] = b[1][1] && b[1][2];
      // east[4] = b[1][4] && (b[1][2] || b[1][5]);
      // east[7] = b[1][7] && (b[1][5] || b[1][8]);
      // east[0] = b[1][0] && east[1];
      // east[3] = b[1][3] && (east[1] || east[4]);
      // east[6] = b[1][6] && (east[4] || east[7]);

      auto white = b[1] & ~(east | west);

      for (auto i = 0; i < 9; ++i) {
        buf[i * cell_states + kWhite - min_cell_state] = white[i];
        buf[i * cell_states + kWhiteEast - min_cell_state] = east[i];
        buf[i * cell_states + kWhiteWest - min_cell_state] = west[i];
      }
    }

    {
      std::bitset<9> north, south;
      // north[0] = b[0][0];
      // north[1] = b[0][1];
      // north[2] = b[0][2];
      // north[3] = b[0][3] && (b[0][0] || b[0][1]);
      // north[4] = b[0][4] && (b[0][1] || b[0][2]);
      // north[5] = b[0][5] && b[0][2];
      // north[6] = b[0][6] && (north[3] || north[4]);
      // north[7] = b[0][7] && (north[4] || north[5]);
      // north[8] = b[0][8] && north[5];

      // south[6] = b[0][6];
      // south[7] = b[0][7];
      // south[8] = b[0][8];
      // south[3] = b[0][3] && b[0][6];
      // south[4] = b[0][4] && (b[0][6] || b[0][7]);
      // south[5] = b[0][5] && (b[0][7] || b[0][8]);
      // south[0] = b[0][0] && south[3];
      // south[1] = b[0][1] && (south[3] || south[4]);
      // south[2] = b[0][2] && (south[4] || south[5]);

      auto black = b[0] & ~(north | south);

      for (auto i = 0; i < 9; ++i) {
        buf[i * cell_states + kBlack - min_cell_state] = black[i];
        buf[i * cell_states + kBlackNorth - min_cell_state] = north[i];
        buf[i * cell_states + kBlackSouth - min_cell_state] = south[i];
      }
    }

    // #ifdef DEBUG
    for (int i = 0; i < 9; ++i) {
      int s = 0;
      for (int j = 0; j < 9; ++j) {
        s += buf[i * cell_states + j];
      }
      assert(s == 1);
    }
    // #endif
  }
};

struct CornerDhState : public DhState<false> {
  CornerDhState() : DhState<false>() {
    x[0][0] = (1 << 1) + 1;
    x[1][8] = (1 << 1) + 1;
    t[0] = 1;
    t[1] = 1;
  }

  uint32_t available_actions() const {
    uint32_t ans = DhState<false>::available_actions();
    return ans & (player() == 0 ? 0b011111111 : 0b111111110);
  }
};