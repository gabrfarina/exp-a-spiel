#pragma once

#include "averager.h"
#include "traverser.h"
#include "utils.h"

struct CfrConf {
  AveragingStrategy avg = AveragingStrategy::QUADRATIC;
  bool alternation = true;
  bool dcfr = true;
  bool rmplus = false;
  bool predictive = false;

  void validate() const {
    CHECK(avg != AveragingStrategy::CUSTOM,
          "CFR averaging strategy cannot be CUSTOM");
    CHECK(!dcfr || !rmplus,
          "dcfr and rmplus options cannot both be set to true");
  }
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
  Real update_regrets_(int p);
  Real update_prediction_(int p);

  CfrConf conf_;
  std::shared_ptr<Traverser<T>> traverser_;
  PerPlayer<Averager> averagers_;

  size_t n_steps_ = 0;
  PerPlayer<std::valarray<Real>> regrets_, bh_;
  std::valarray<Real> gradient_copy_;
};