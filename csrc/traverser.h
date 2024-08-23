#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "dh_state.h"
#include "log.h"
#include "pttt_state.h"

using Real = double;

struct InfosetMetadata {
  uint32_t legal_actions;
  uint32_t infoset_id;

  bool operator==(const InfosetMetadata &other) const {
    return legal_actions == other.legal_actions &&
           infoset_id == other.infoset_id;
  }
};

// Maps from infoset to legal action mask
using InfosetMap = std::unordered_map<uint64_t, InfosetMetadata>;

struct Treeplex {
  InfosetMap infosets;
  std::vector<uint64_t> infoset_keys;
  std::vector<uint32_t> legal_actions;
  std::vector<uint32_t> parent_index;

  uint32_t num_infosets() const { return infoset_keys.size(); }
  void validate_vector(const Real *buf) const;
  void validate_strategy(const Real *buf) const;
  void set_uniform(Real *buf) const;
  void bh_to_sf(Real *buf) const;
  Real br(Real *grad, Real *strat = nullptr) const;
};

struct EvExpl {
  Real ev0;
  // gradient of utility wrt the player strategies
  std::array<std::valarray<Real>, 2> gradient;
  // expl[0] is how exploitable player 0 is by a best-responding player 1
  std::array<Real, 2> expl;
  // best_response[0] is the best response to player 1's strategy
  std::array<std::valarray<Real>, 2> best_response;
};

template <typename T> struct Traverser {
  Treeplex treeplex[2];
  std::valarray<Real> gradients[2];

  Traverser();
  void compute_gradients(const std::array<const Real *, 2> strategies);
  EvExpl ev_and_exploitability(const std::array<const Real *, 2> strategies);

private:
  std::valarray<Real> bufs_[2][9];
  std::valarray<Real> sf_strategies_[2];

  void compute_sf_strategies_(const std::array<const Real *, 2> strategies) {
#pragma omp parallel for
    for (int p = 0; p < 2; ++p) {
      memcpy(&sf_strategies_[p][0], strategies[p],
             treeplex[p].num_infosets() * 9 * sizeof(Real));
      treeplex[p].bh_to_sf(&sf_strategies_[p][0]);
    }
  }
};