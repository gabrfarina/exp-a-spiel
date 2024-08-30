#pragma once
#include <array>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstdint>
#include <span>
#include <valarray>
#include <vector>

#include "averager.h"
#include "types.h"

struct InfosetMetadata {
  uint32_t legal_actions;
  uint32_t infoset_id;

  bool operator==(const InfosetMetadata &other) const {
    return legal_actions == other.legal_actions &&
           infoset_id == other.infoset_id;
  }
};

// Maps from infoset to legal action mask
using InfosetMap = boost::unordered_flat_map<uint64_t, InfosetMetadata>;
void relu_noramlize(RealBuf buf, const uint32_t mask);
Real dot(ConstRealBuf a, ConstRealBuf b);
struct Treeplex {
  InfosetMap infosets;
  std::vector<uint64_t> infoset_keys;
  std::vector<uint32_t> legal_actions;
  std::vector<uint32_t> parent_index;

  uint32_t num_infosets() const { return infoset_keys.size(); }
  void validate_vector(ConstRealBuf buf) const;
  void validate_strategy(ConstRealBuf buf) const;
  void set_uniform(RealBuf buf) const;
  void bh_to_sf(RealBuf buf) const;
  void sf_to_bh(RealBuf buf) const;
  Real br(RealBuf grad, RealBuf strat = std::span<Real>()) const;
  // inplace relu + renomalize, adds a small epsilon for stability (numeric and strategic) to the regrets
  void regret_to_bh(RealBuf buf) const;
};

struct EvExpl {
  Real ev0;
  // gradient of utility wrt the player strategies
  PerPlayer<std::valarray<Real>> gradient;
  // expl[0] is how exploitable player 0 is by a best-responding player 1
  PerPlayer<Real> expl;
  // best_response[0] is the best response to player 1's strategy
  PerPlayer<std::valarray<Real>> best_response;
};

template <typename T> struct Traverser {
  PerPlayer<std::shared_ptr<Treeplex>> treeplex;
  PerPlayer<std::valarray<Real>> gradients;
  Traverser();

  void compute_gradients(const PerPlayer<ConstRealBuf> strategies);
  EvExpl ev_and_exploitability(const PerPlayer<ConstRealBuf> strategies);
  Averager new_averager(const uint8_t player);

private:
  PerPlayer<std::array<std::valarray<Real>, 9>> bufs_;
  PerPlayer<std::valarray<Real>> sf_strategies_;

  void compute_sf_strategies_(const PerPlayer<ConstRealBuf> strategies);
};
