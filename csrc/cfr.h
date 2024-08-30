#pragma once

#include "averager.h"
#include "traverser.h"
#include "types.h"

struct CfrConf {
  AVERAGING_STRATEGY avg = AVERAGING_STRATEGY::LINEAR;
  bool alternation = true;
  bool dcfr = false;
  bool rmplus = false;
  bool pcfrp = false;
};

template <typename T>
class CfrSolver {
 public:
  CfrSolver(std::shared_ptr<Traverser<T>> traverser, const CfrConf conf);

  void step();
  ConstRealBuf get_regrets(const uint8_t player) const {
    return regrets_[player];
  }
  ConstRealBuf get_bh(const uint8_t player) const { return bh_[player]; }

 private:
  // does not update the gradient
  void inner_step();
  Real update_regrets(int p);
  Real update_regrets_pcfrp(int p);
  CfrConf conf_;
  std::shared_ptr<Traverser<T>> traverser_;
  PerPlayer<Averager> averagers_;

  size_t n_iters_ = 0;
  PerPlayer<std::valarray<Real>> regrets_, bh_;
};