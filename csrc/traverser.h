#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "log.h"

using Real = float;
const uint32_t NUM_INFOS_PL1 = 3720850;
const uint32_t NUM_INFOS_PL2 = 2352067;

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
  Real br_value(Real *buf) const;
};

struct EvExpl {
  Real ev;
  std::array<Real, 2> expl;
};

struct DhTraverser {
  Treeplex treeplex[2];
  std::valarray<Real> gradients[2];

  DhTraverser();
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