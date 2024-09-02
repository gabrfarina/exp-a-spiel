#include "cfr.h"

#include <limits>
#include <valarray>

#include "dh_state.h"
#include "pttt_state.h"
#include "traverser.h"
#include "utils.h"

template <typename T>
CfrSolver<T>::CfrSolver(std::shared_ptr<Traverser<T>> traverser,
                        const CfrConf conf)
    : conf_(conf), traverser_(traverser),
      averagers_{traverser_->new_averager(0, conf.avg),
                 traverser_->new_averager(1, conf.avg)},
      regrets_{
          std::valarray<Real>(0., traverser_->treeplex[0]->num_infosets() * 9),
          std::valarray<Real>(0., traverser_->treeplex[1]->num_infosets() * 9)},
      bh_{regrets_} {
  conf_.validate();

  for (auto p : {0, 1}) {
    traverser_->treeplex[p]->set_uniform(bh_[p]);
    averagers_[p].push(bh_[p]);
  }

  n_steps_ = 0;
}

template <typename T> void CfrSolver<T>::step() {
  traverser_->compute_gradients({bh_[0], bh_[1]});
  inner_step_();
  if (!conf_.alternation) {
    inner_step_();
  }
}

template <typename T> void CfrSolver<T>::inner_step_() {
  constexpr double dcfr_alpha = 1.5;
  constexpr double dcfr_beta = 0.0;
  const auto p = n_steps_ % 2;
  const auto p_iters = (n_steps_ / 2) + 1;

  if (conf_.predictive)
    gradient_copy_ = traverser_->gradients[p];

  update_regrets_(p);
  if (conf_.predictive) {
    update_prediction_(p);
  } else {
    bh_[p] = regrets_[p];
    traverser_->treeplex[p]->regret_to_bh(bh_[p]);
  }

  assert(traverser_->treeplex[p]->is_valid_strategy(bh_[p]));
  // + 1 so we take into account the initial uniform strategy
  averagers_[p].push(bh_[p]);

  const Real neg_discount =
      conf_.rmplus ? 0.
      : conf_.dcfr ? (1. - 1. / (1. + std::pow((Real)p_iters, dcfr_beta)))
                   : 1.;
  const Real pos_discount =
      conf_.dcfr ? (1. - 1. / (1. + std::pow((Real)p_iters, dcfr_alpha))) : 1.;

  INFO("pos_discount: %f, neg_discount: %f", pos_discount, neg_discount);

  for (auto &i : regrets_[p])
    if (i > 0)
      i *= pos_discount;
    else
      i *= neg_discount;

  ++n_steps_;
}

template <typename T> Real CfrSolver<T>::update_prediction_(int p) {
  assert(traverser_->treeplex[p]->is_valid_strategy(bh_[p]));
  assert(traverser_->treeplex[p]->is_valid_vector(gradient_copy_));

  Real ev = 0;
  for (int32_t i = traverser_->treeplex[p]->num_infosets() - 1; i >= 0; --i) {
    const uint64_t info = traverser_->treeplex[p]->infoset_keys[i];
    const uint32_t mask = traverser_->treeplex[p]->legal_actions[i];

    ev = dot(std::span(gradient_copy_).subspan(i * 9, 9),
             std::span(bh_[p]).subspan(i * 9, 9));

    for (uint32_t j = 0; j < 9; ++j) {
      if (is_valid(mask, j)) {
        bh_[p][i * 9 + j] =
            regrets_[p][i * 9 + j] + gradient_copy_[i * 9 + j] - ev;
      }
    }
    relu_normalize(std::span(bh_[p]).subspan(i * 9, 9), mask);

    ev = dot(std::span(gradient_copy_).subspan(i * 9, 9),
             std::span(bh_[p]).subspan(i * 9, 9));

    if (i) {
      const uint32_t parent = traverser_->treeplex[p]->parent_index[i];
      const uint32_t parent_a = parent_action(info);
      gradient_copy_[parent * 9 + parent_a] += ev;
    }
  }

  assert(traverser_->treeplex[p]->is_valid_strategy(bh_[p]));
  assert(traverser_->treeplex[p]->is_valid_vector(gradient_copy_));

  return ev;
}

template <typename T> Real CfrSolver<T>::update_regrets_(int p) {
  assert(traverser_->treeplex[p]->is_valid_strategy(bh_[p]));
  assert(traverser_->treeplex[p]->is_valid_vector(regrets_[p]));
  assert(traverser_->treeplex[p]->is_valid_vector(traverser_->gradients[p]));

  Real ev = 0;
  for (int32_t i = traverser_->treeplex[p]->num_infosets() - 1; i >= 0; --i) {
    const uint64_t info = traverser_->treeplex[p]->infoset_keys[i];
    const uint32_t mask = traverser_->treeplex[p]->legal_actions[i];

    ev = dot(std::span(traverser_->gradients[p]).subspan(i * 9, 9),
             std::span(bh_[p]).subspan(i * 9, 9));
    for (uint32_t j = 0; j < 9; ++j) {
      if (is_valid(mask, j)) {
        regrets_[p][i * 9 + j] += traverser_->gradients[p][i * 9 + j] - ev;
      }
    }

    if (i) {
      const uint32_t parent = traverser_->treeplex[p]->parent_index[i];
      const uint32_t parent_a = parent_action(info);
      traverser_->gradients[p][parent * 9 + parent_a] += ev;
    }
  }

  assert(traverser_->treeplex[p]->is_valid_strategy(bh_[p]));
  assert(traverser_->treeplex[p]->is_valid_vector(regrets_[p]));
  assert(traverser_->treeplex[p]->is_valid_vector(traverser_->gradients[p]));

  return ev;
}

template class CfrSolver<DhState<false>>;
template class CfrSolver<DhState<true>>;
template class CfrSolver<CornerDhState>;
template class CfrSolver<PtttState<false>>;
template class CfrSolver<PtttState<true>>;