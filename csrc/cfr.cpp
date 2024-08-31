#include "cfr.h"

#include <limits>

#include "dh_state.h"
#include "pttt_state.h"
#include "traverser.h"
#include "utils.h"

template <typename T>
CfrSolver<T>::CfrSolver(std::shared_ptr<Traverser<T>> traverser,
                        const CfrConf conf)
    : conf_(conf),
      traverser_(traverser),
      averagers_{traverser_->new_averager(0), traverser_->new_averager(1)},
      regrets_{
          std::valarray<Real>(0., traverser_->treeplex[0]->num_infosets() * 9),
          std::valarray<Real>(0., traverser_->treeplex[1]->num_infosets() * 9)},
      bh_{regrets_} {
  conf_.validate();

  for (auto p : {0, 1}) {
    traverser_->treeplex[p]->set_uniform(bh_[p]);
    averagers_[p].push(bh_[p], 1.);
  }

  n_iters_ = 2;
}

template <typename T>
void CfrSolver<T>::step() {
  traverser_->compute_gradients({bh_[0], bh_[1]});
  inner_step_();
  if (!conf_.alternation) {
    inner_step_();
  }
}

template <typename T>
void CfrSolver<T>::inner_step_() {
  constexpr double dcfr_alpha = 0.5;
  constexpr double dcfr_beta = 1.0;
  const auto p = n_iters_ % 2;
  const auto p_iters = n_iters_ / 2;

  if (conf_.pcfrp) {
    update_regrets_<true>(p);
  } else {
    update_regrets_<false>(p);
  }
  traverser_->treeplex[p]->validate_strategy(bh_[p]);
  // + 1 so we take into account the initial uniform strategy
  averagers_[p].push(bh_[p], iter_weight(conf_.avg, p_iters + 1));

  const Real neg_discount = conf_.rmplus ? 0.
                            : conf_.dcfr
                                ? (1. - (1. + std::pow(p_iters, dcfr_beta)))
                                : 1.;
  const Real pos_discount =
      conf_.dcfr ? (1. - 1. / (1. + std::pow(p_iters, dcfr_alpha))) : 1.;

  for (auto &i : regrets_[p])
    if (i > 0)
      i *= pos_discount;
    else
      i *= neg_discount;
  ++n_iters_;
}

template <typename T>
template <bool predictive>
Real CfrSolver<T>::update_regrets_(int p) {
  traverser_->treeplex[p]->validate_strategy(bh_[p]);
  traverser_->treeplex[p]->validate_vector(regrets_[p]);
  traverser_->treeplex[p]->validate_vector(traverser_->gradients[p]);

  Real ev = 0;
  for (int32_t i = traverser_->treeplex[p]->num_infosets() - 1; i >= 0; --i) {
    const uint64_t info = traverser_->treeplex[p]->infoset_keys[i];
    const uint32_t mask = traverser_->treeplex[p]->legal_actions[i];

    Real max_val = std::numeric_limits<Real>::lowest();

    for (uint32_t j = 0; j < 9; ++j) {
      if (is_valid(mask, j) &&
          (traverser_->gradients[p][i * 9 + j] > max_val)) {
        max_val = traverser_->gradients[p][i * 9 + j];
      }
    }
    ev = dot(std::span(traverser_->gradients[p]).subspan(i * 9, 9),
             std::span(bh_[p]).subspan(i * 9, 9));
    for (uint32_t j = 0; j < 9; ++j) {
      if (is_valid(mask, j)) {
        regrets_[p][i * 9 + j] += traverser_->gradients[p][i * 9 + j] - ev;
        bh_[p][i * 9 + j] = regrets_[p][i * 9 + j];
      }
    }
    relu_normalize(std::span(bh_[p]).subspan(i * 9, 9), mask);
    if constexpr (predictive)
      ev = dot(std::span(traverser_->gradients[p]).subspan(i * 9, 9),
               std::span(bh_[p]).subspan(i * 9, 9));

    if (i) {
      const uint32_t parent = traverser_->treeplex[p]->parent_index[i];
      const uint32_t parent_a = parent_action(info);
      traverser_->gradients[p][parent * 9 + parent_a] += ev;
    }
  }

  traverser_->treeplex[p]->validate_strategy(bh_[p]);
  traverser_->treeplex[p]->validate_vector(regrets_[p]);
  traverser_->treeplex[p]->validate_vector(traverser_->gradients[p]);

  return ev;
}

template class CfrSolver<DhState<false>>;
template class CfrSolver<DhState<true>>;
template class CfrSolver<CornerDhState>;
template class CfrSolver<PtttState<false>>;
template class CfrSolver<PtttState<true>>;