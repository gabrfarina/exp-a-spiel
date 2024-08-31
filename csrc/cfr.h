#pragma once

#include "averager.h"
#include "traverser.h"
#include "utils.h"

struct CfrConf {
  AveragingStrategy avg = AveragingStrategy::LINEAR;
  bool alternation = true;
  bool dcfr = false;
  bool rmplus = false;
  bool pcfrp = false;
};

template <typename T> class CfrSolver {
public:
  CfrSolver(std::shared_ptr<Traverser<T>> traverser, const CfrConf conf);

  void step();

  ConstRealBuf get_regrets(const uint8_t player) const {
    return regrets_[player];
  }
  ConstRealBuf get_bh(const uint8_t player) const { return bh_[player]; }
  std::valarray<Real> get_avg_bh(const uint8_t player) const {
    return averagers_[player].running_avg();
  }

private:
  void inner_step_(); // Warning: this does not update the gradient
  template <bool predictive> Real update_regrets_(int p);

  CfrConf conf_;
  std::shared_ptr<Traverser<T>> traverser_;
  PerPlayer<Averager> averagers_;

  size_t n_iters_ = 0;
  PerPlayer<std::valarray<Real>> regrets_, bh_;
};