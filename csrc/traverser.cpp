#include "traverser.h"

#include <cmath>
#include <cstring>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "dh_state.h"
#include "log.h"
#include "pttt_state.h"

using std::size_t;

namespace {
template <typename T>
uint64_t discover_infosets_thread(T root, PerPlayer<InfosetMap> *infosets) {
  T stack[100];
  stack[0] = root;
  size_t stack_len = 1;
  uint64_t count = 0;

  while (stack_len) {
    // Pop from stack
    const T s = stack[--stack_len];
    const uint8_t p = s.player();
    ++count;

    if (s.winner() == 0xff) {
      uint32_t a = s.available_actions();
      const uint64_t info = s.get_infoset();
      const InfosetMetadata md{
          .legal_actions = a,
          .infoset_id = UINT32_MAX,
      };
      assert(!(*infosets)[p].count(info) || (*infosets)[p][info] == md);
      (*infosets)[p][info] = md;

      for (int i = 0; i < 9; ++i, a >>= 1) {
        if (a & 1) {
          T ss = s;
          ss.next(i);
          stack[stack_len++] = ss;
          assert(stack_len < 100);
        }
      }
    }
  }

  return count;
}

template <typename T>
void compute_gradients_thread(
    T root, const PerPlayer<uint32_t> init_parent_seqs,
    const PerPlayer<std::shared_ptr<Treeplex>> treeplex,
    PerPlayer<ConstRealBuf> sf_strategies, PerPlayer<RealBuf> gradients) {
  std::tuple<T,                  // Current state
             PerPlayer<uint32_t> // Parent seqs
             >
      stack[100];
  stack[0] = {root, init_parent_seqs};
  size_t stack_len = 1;

  while (stack_len) {
    const auto it = stack[--stack_len];
    const T &s = std::get<0>(it);
    const uint8_t p = s.player();
    const uint8_t w = s.winner();
    const uint64_t infoset = s.get_infoset();
    const PerPlayer<uint32_t> &seqs = std::get<1>(it);

    if (w == 0xff) {
      const uint32_t info_id = treeplex[p]->infosets.at(infoset).infoset_id;
      PerPlayer<uint32_t> new_seqs = seqs;

      uint32_t a = s.available_actions();
      for (uint32_t i = 0; a; ++i, a >>= 1) {
        if (a & 1) {
          T ss = s;
          ss.next(i);
          new_seqs[p] = 9 * info_id + i;
          stack[stack_len++] = {ss, new_seqs};
        }
      }
    } else if (w == 0 || w == 1) {
      const Real sign = -2.0 * w + 1.0; // 1 if w == 0 else -1
      gradients[0][seqs[0]] += sign * sf_strategies[1][seqs[1]];
      gradients[1][seqs[1]] -= sign * sf_strategies[0][seqs[0]];
    }
  }
}
} // namespace

void Treeplex::validate_vector(ConstRealBuf buf) const {
  CHECK(buf.size() == num_infosets() * 9,
        "Vector size mismatch (expected %d, found %ld)", num_infosets() * 9,
        buf.size());

  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    for (uint32_t j = 0; j < 9; ++j) {
      if (!(a & (1 << j))) {
        CHECK(buf[i * 9 + j] == 0, "strategy must be zero for illegal actions");
      }
    }
  }
}

void Treeplex::validate_strategy(ConstRealBuf buf) const {
  CHECK(buf.size() == num_infosets() * 9,
        "Strategy size mismatch (expected %d, found %ld)", num_infosets() * 9,
        buf.size());

  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    Real sum = 0;
    for (uint32_t j = 0; j < 9; ++j) {
      CHECK(buf[i * 9 + j] >= 0, "strategy must be nonnegative, was %f",
            buf[i * 9 + j]);
      CHECK(buf[i * 9 + j] <= 1, "strategy must be at most 1, was %f",
            buf[i * 9 + j]);

      if (a & (1 << j)) {
        sum += buf[i * 9 + j];
      } else {
        CHECK(buf[i * 9 + j] == 0, "strategy must be zero for illegal actions");
      }
    }
    CHECK(std::abs(sum - 1.0) < 1e-6, "strategy must sum to 1 but found %f",
          sum);
  }
}

void Treeplex::set_uniform(RealBuf buf) const {
  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    const uint8_t na = __builtin_popcount(a);

    for (uint32_t j = 0; j < 9; ++j) {
      buf[i * 9 + j] = Real(!!(a & (1 << j))) / na;
    }
  }

#ifdef DEBUG
  validate_strategy(buf);
#endif
}

void Treeplex::bh_to_sf(RealBuf buf) const {
#ifdef DEBUG
  validate_strategy(buf);
#endif

  for (uint32_t i = 1; i < num_infosets(); ++i) {
    const uint64_t info = infoset_keys[i];
    const uint32_t parent = parent_index[i];
    const uint32_t parent_a = parent_action(info);
    const Real parent_prob = buf[parent * 9 + parent_a];

    for (uint32_t j = 0; j < 9; ++j) {
      buf[i * 9 + j] *= parent_prob;
    }
  }

#ifdef DEBUG
  validate_vector(buf);
#endif
}

constexpr Real bh_eps = 1e-6;
void Treeplex::sf_to_bh(RealBuf buf) const {
#ifdef DEBUG
  validate_vector(buf);
#endif
  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    Real s = 0;
    for (uint32_t j = 0; j < 9; ++j) {
      s += (buf[i * 9 + j] + bh_eps) * (a & (1 << j));
    }
    for (uint32_t j = 0; j < 9; ++j) {
      buf[i * 9 + j] = (buf[i * 9 + j] + bh_eps) / s * (a & (1 << j));
    }
  }
}

Real Treeplex::br(RealBuf buf, RealBuf strat) const {
  std::fill(strat.begin(), strat.end(), 0.0);

  validate_vector(buf);
  if (!strat.empty())
    validate_vector(buf);

  Real max_val = std::numeric_limits<Real>::lowest();
  for (int32_t i = num_infosets() - 1; i >= 0; --i) {
    const uint64_t info = infoset_keys[i];
    const uint32_t mask = legal_actions[i];

    max_val = std::numeric_limits<Real>::lowest();
    uint8_t best_action = 0xff;
    for (uint32_t j = 0; j < 9; ++j) {
      if ((mask & (1 << j)) && (buf[i * 9 + j] > max_val)) {
        best_action = j;
        max_val = buf[i * 9 + j];
      }
    }
    assert(best_action != 0xff);

    if (i) {
      const uint32_t parent = parent_index[i];
      const uint32_t parent_a = parent_action(info);
      buf[parent * 9 + parent_a] += max_val;
    }
    if (!strat.empty()) {
      strat[i * 9 + best_action] = 1.0;
    }
  }

#ifdef DEBUG
  if (!strat.empty())
    validate_strategy(strat);
#endif

  return max_val;
}

void Treeplex::regret_to_bh(RealBuf buf) const {
  constexpr Real SMALL = 1e-10;

#ifdef DEBUG
  validate_vector(buf);
#endif
  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    Real s = 0;
    for (uint32_t j = 0; j < 9; ++j) {
      auto const x = (std::max<Real>(buf[i * 9 + j], 0)) * (a & (1 << j));
      s += x;
      buf[i * 9 + j] = x;
    }
    if (s < SMALL) {
      s = 0;
      for (uint32_t j = 0; j < 9; ++j) {
        const auto x = a & (1 << j);
        buf[i * 9 + j] = x;
        s += x;
      }
    }
    for (uint32_t j = 0; j < 9; ++j) {
      buf[i * 9 + j] /= s;
    }
  }

#ifdef DEBUG
  validate_strategy(buf);
#endif
}

template <typename T> Traverser<T>::Traverser() {
  treeplex[0]->infosets.reserve(5000000);
  treeplex[1]->infosets.reserve(5000000);

  treeplex[0]->infosets[0] =
      InfosetMetadata{.legal_actions = 0b111111111, .infoset_id = UINT32_MAX};
  treeplex[1]->infosets[0] =
      InfosetMetadata{.legal_actions = 0b111111111, .infoset_id = UINT32_MAX};

  INFO("discovering infosets (num threads: %d)...", omp_get_max_threads());
  uint64_t count = 10;
#pragma omp parallel for reduction(+ : count)
  for (int i = 0; i < 9 * 9; ++i) {
    T s{};
    {
      const uint8_t a = i % 9;
      assert(s.available_actions() & (1 << a));
      s.next(a);
    }
    {
      const uint8_t a = i / 9;
      assert(s.available_actions() & (1 << a));
      s.next(a);
    }

    PerPlayer<InfosetMap> thread_infosets;
    const uint64_t thread_count =
        ::discover_infosets_thread(s, &thread_infosets);
    count += thread_count;

    INFO("  > thread %02d found %.2fM infosets (%.2fB nodes)", i,
         (thread_infosets[0].size() + thread_infosets[1].size()) / 1e6,
         thread_count / 1e9);

#pragma omp critical
    {
      treeplex[0]->infosets.insert(thread_infosets[0].begin(),
                                   thread_infosets[0].end());
      treeplex[1]->infosets.insert(thread_infosets[1].begin(),
                                   thread_infosets[1].end());
    }
  }
  INFO("... discovery terminated. Found %.2fM infosets across %.2fB nodes",
       (treeplex[0]->infosets.size() + treeplex[1]->infosets.size()) / 1e6,
       count / 1e9);

#ifdef DEBUG
  for (int p = 0; p < 2; ++p) {
    INFO("checking infosets of player %d...", p + 1);
    for (const auto &it : treeplex[p]->infosets) {
      const uint64_t infoset = it.first;
      if (infoset) {
        const uint64_t parent = parent_infoset(infoset);
        CHECK(parent <= infoset, "Parent infoset is greater than child");
        CHECK(treeplex[p]->infosets.count(parent),
              "Parent infoset %ld of %ld not found", parent, infoset);
        CHECK(treeplex[p]->infosets[parent].legal_actions &
                  (1u << parent_action(infoset)),
              "Parent action is illegal");
      }
    }
  }
  INFO("... all infosets are consistent");
#endif

#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    treeplex[p]->infoset_keys.reserve(treeplex[p]->infosets.size());
    treeplex[p]->parent_index.resize(treeplex[p]->infosets.size());
    treeplex[p]->legal_actions.resize(treeplex[p]->infosets.size());
    treeplex[p]->parent_index[0] = UINT32_MAX;

    for (auto &it : treeplex[p]->infosets) {
      treeplex[p]->infoset_keys.push_back(it.first);
    }
    INFO("sorting infoset and assigning indices for player %d...", p + 1);
    std::sort(treeplex[p]->infoset_keys.begin(),
              treeplex[p]->infoset_keys.end());
    for (uint32_t i = 0; i < treeplex[p]->infoset_keys.size(); ++i) {
      const uint64_t infoset = treeplex[p]->infoset_keys[i];
      assert(treeplex[p]->infosets.count(infoset));

      auto &info = treeplex[p]->infosets[infoset];
      info.infoset_id = i;
      if (i) {
        treeplex[p]->parent_index[i] =
            treeplex[p]->infosets[parent_infoset(infoset)].infoset_id;
      }
      treeplex[p]->legal_actions[i] = info.legal_actions;
    }
  }
  assert(*treeplex[0].infoset_keys.begin() == 0);
  assert(*treeplex[1].infoset_keys.begin() == 0);
  assert(treeplex[0].infoset_keys.size() == treeplex[0].infosets.size() &&
         treeplex[0].parent_index.size() == treeplex[0].infosets.size());
  assert(treeplex[1].infoset_keys.size() == treeplex[1].infosets.size() &&
         treeplex[1].parent_index.size() == treeplex[1].infosets.size());

  for (int i = 0; i < 9; ++i) {
    bufs_[0][i].resize(treeplex[0]->num_infosets() * 9);
    bufs_[1][i].resize(treeplex[1]->num_infosets() * 9);
  }
  gradients[0].resize(treeplex[0]->num_infosets() * 9, 0.0);
  gradients[1].resize(treeplex[1]->num_infosets() * 9, 0.0);
  sf_strategies_[0].resize(treeplex[0]->num_infosets() * 9, 0.0);
  sf_strategies_[1].resize(treeplex[1]->num_infosets() * 9, 0.0);

  INFO("... all done.");
}

template <typename T>
void Traverser<T>::compute_gradients(const PerPlayer<ConstRealBuf> strategies) {
  INFO("begin gradient computation (num threads: %d)...",
       omp_get_max_threads());
  treeplex[0]->validate_strategy(strategies[0]);
  treeplex[0]->validate_vector(gradients[0]);
  treeplex[1]->validate_strategy(strategies[1]);
  treeplex[1]->validate_vector(gradients[1]);
  compute_sf_strategies_(strategies);

  gradients[0] = 0.0;
  gradients[1] = 0.0;

  for (int i = 0; i < 9; ++i) {
    bufs_[0][i] = 0.0;
    bufs_[1][i] = 0.0;
  }

  uint32_t num_finished = 0;
#pragma omp parallel for
  for (unsigned i = 0; i < 9 * 9; ++i) {
    T s{};
    assert(!s.get_infoset());
    s.next(i % 9); // pl1's move
    assert(!s.get_infoset());
    s.next(i / 9); // pl2's move

    const PerPlayer<uint32_t> parent_seqs = {i % 9, i / 9};
    const PerPlayer<RealBuf> thread_gradients = {bufs_[0][i / 9],
                                                 bufs_[1][i % 9]};
    ::compute_gradients_thread(s, parent_seqs, treeplex,
                               {sf_strategies_[0], sf_strategies_[1]},
                               thread_gradients);

#pragma omp critical
    {
      ++num_finished;
      if (num_finished % 10 == 0) {
        INFO("  > %2d/81 threads returned", num_finished);
      }
    }
  }

  INFO("... aggregating thread buffers...");

#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    for (int j = 0; j < 9; ++j)
      gradients[p] += bufs_[p][j];
  }

  INFO("... all done.");
}

template <typename T>
EvExpl
Traverser<T>::ev_and_exploitability(const PerPlayer<ConstRealBuf> strategies) {
  EvExpl out;

  INFO("begin exploitability computation...");
  compute_gradients(strategies);
  out.gradient[0] = gradients[0];
  out.gradient[1] = gradients[1];

  INFO("computing expected value...");
  Real ev0 = 0.0;
  for (uint32_t i = 0; i < treeplex[0]->num_infosets() * 9; ++i) {
    ev0 += sf_strategies_[0][i] * gradients[0][i];
  }
  out.ev0 = ev0;

#ifdef DEBUG
  {
    INFO("double checking expected value...");

    Real ev1 = 0.0;
    for (uint32_t i = 0; i < treeplex[1]->num_infosets() * 9; ++i) {
      ev1 += sf_strategies_[1][i] * gradients[1][i];
    }

    CHECK(std::abs(ev0 + ev1) < 1e-3, "Expected values differ: %.6f != %.6f",
          ev0, ev1);
  }
#endif

  out.expl = {ev0, -ev0};
  out.best_response[0].resize(treeplex[0]->num_infosets() * 9, 0.0);
  out.best_response[1].resize(treeplex[1]->num_infosets() * 9, 0.0);

  INFO("computing exploitabilities...");
#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    out.expl[1 - p] += treeplex[p]->br(gradients[p], out.best_response[p]);
  }

  INFO("... all done. (ev0 = %.6f, expl = %.6f, %.6f)", ev0, out.expl[0],
       out.expl[1]);
  return out;
}

template <typename T>
Averager Traverser<T>::new_averager(const uint8_t player) {
  CHECK(player == 0 || player == 1, "Invalid player %d", player);
  return Averager(treeplex[player]);
}

template <typename T>
void Traverser<T>::compute_sf_strategies_(
    const PerPlayer<ConstRealBuf> strategies) {
#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    std::copy(strategies[p].begin(), strategies[p].end(),
              std::begin(sf_strategies_[p]));
    treeplex[p]->bh_to_sf(sf_strategies_[p]);
  }
}

template struct Traverser<DhState<false>>;
template struct Traverser<DhState<true>>;
template struct Traverser<PtttState<false>>;
template struct Traverser<PtttState<true>>;