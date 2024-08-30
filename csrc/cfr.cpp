#include "cfr.h"
#include "dh_state.h"
#include "pttt_state.h"
#include "traverser.h"

template <typename T>
CfrSolver<T>::CfrSolver(std::shared_ptr<Traverser<T>> traverser,
                        const CfrConf conf)
    : conf_(conf), traverser_(traverser),
      averagers_{traverser_->new_averager(0), traverser_->new_averager(1)},
      regrets_{
          std::valarray<Real>(0., traverser_->treeplex[0]->num_infosets() * 9),
          std::valarray<Real>(0., traverser_->treeplex[1]->num_infosets() * 9)},
      bh_{regrets_} {
  traverser_->treeplex[0]->set_uniform(bh_[0]);
  traverser_->treeplex[1]->set_uniform(bh_[0]);
  averagers_[0].push(bh_[0], 1);
  averagers_[1].push(bh_[0], 1);
}

template <typename T> void CfrSolver<T>::step() {
  traverser_->compute_gradients({bh_[0], bh_[1]});
  inner_step();
  if (!conf_.alternation) {
    inner_step();
  }
}
template <typename T> void CfrSolver<T>::inner_step() {
  ++n_iters_;
  const auto p = n_iters_ % 2;
  const auto p_iters = n_iters_ / 2;
  if (conf_.pcfrp) {
    update_regrets_pcfrp(p);
  } else {
    update_regrets(p);
    std::copy(std::begin(regrets_[p]), std::begin(regrets_[p]),
              std::begin(bh_[p]));
    traverser_->treeplex[p]->regret_to_bh(bh_[p]);
  }
  traverser_->treeplex[p]->validate_strategy(bh_[p]);
  averagers_[p].push(bh_[p], iter_weight(conf_.avg, p_iters));
  Real neg_discount = conf_.rmplus ? 0.0 : conf_.dcfr ? 0.5 : 1.0;
  Real pos_discount = 1.0;
  for (auto &i : regrets_[p])
    if (i > 0)
      i *= pos_discount;
    else
      i *= neg_discount;
}

template <typename T> Real CfrSolver<T>::update_regrets_pcfrp(int p) {
#ifdef DEBUG
  validate_strategy(bf[p]);
  validate_vector(regrets[p]);
  validate_vector(traverser_->gradient[p]);
#endif

  Real ev = 0;
  for (int32_t i = traverser_->treeplex[p]->num_infosets() - 1; i >= 0; --i) {
    const uint64_t info = traverser_->treeplex[p]->infoset_keys[i];
    const uint32_t mask = traverser_->treeplex[p]->legal_actions[i];

    Real max_val = std::numeric_limits<Real>::lowest();

    for (uint32_t j = 0; j < 9; ++j) {
      if ((mask & (1 << j)) &&
          (traverser_->gradients[p][i * 9 + j] > max_val)) {
        max_val = traverser_->gradients[p][i * 9 + j];
      }
    }

    
    for (uint32_t j = 0; j < 9; ++j) {
      if (mask & (1 << j)) {
        regrets_[p][i * 9 + j] += traverser_->gradients[p][i * 9 + j] - max_val;
        bh_[p][i * 9 + j] = regrets_[p][i * 9 + j];
      }
    }
    relu_noramlize(std::span(bh_[p]).subspan(i * 9, 9), mask);
    ev = dot(std::span(traverser_->gradients[p]).subspan(i * 9, 9),
             std::span(bh_[p]).subspan(i * 9, 9));
             
    if (i) {
      const uint32_t parent = traverser_->treeplex[p]->parent_index[i];
      const uint32_t parent_a = parent_action(info);
      traverser_->gradients[p][parent * 9 + parent_a] += ev;
    }
  }

#ifdef DEBUG
  validate_strategy(bf[p]);
  validate_vector(regrets[p]);
  validate_vector(traverser_->gradient[p]);
#endif

  return ev;
}

template <typename T> Real CfrSolver<T>::update_regrets(int p) {
#ifdef DEBUG
  validate_strategy(bf[p]);
  validate_vector(regrets[p]);
  validate_vector(traverser_->gradient[p]);
#endif

  Real ev = 0;
  for (int32_t i = traverser_->treeplex[p]->num_infosets() - 1; i >= 0; --i) {
    const uint64_t info = traverser_->treeplex[p]->infoset_keys[i];
    const uint32_t mask = traverser_->treeplex[p]->legal_actions[i];

    Real max_val = std::numeric_limits<Real>::lowest();

    for (uint32_t j = 0; j < 9; ++j) {
      if ((mask & (1 << j)) &&
          (traverser_->gradients[p][i * 9 + j] > max_val)) {
        max_val = traverser_->gradients[p][i * 9 + j];
      }
    }

    ev = 0;
    for (uint32_t j = 0; j < 9; ++j) {
      if (mask & (1 << j)) {
        ev += traverser_->gradients[p][i * 9 + j] * bh_[p][i * 9 + j];
        regrets_[p][i * 9 + j] += traverser_->gradients[p][i * 9 + j] - max_val;
      }
    }

    if (i) {
      const uint32_t parent = traverser_->treeplex[p]->parent_index[i];
      const uint32_t parent_a = parent_action(info);
      traverser_->gradients[p][parent * 9 + parent_a] += ev;
    }
  }

#ifdef DEBUG
  validate_strategy(bf[p]);
  validate_vector(regrets[p]);
  validate_vector(traverser_->gradient[p]);
#endif

  return ev;
}

template class CfrSolver<DhState<false>>;
template class CfrSolver<DhState<true>>;
template class CfrSolver<PtttState<false>>;
template class CfrSolver<PtttState<true>>;