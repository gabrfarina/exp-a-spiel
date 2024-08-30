#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sys/types.h>

#include <cstdint>
#include <valarray>

#include "averager.h"
#include "cfr.h"
#include "dh_state.h"
#include "log.h"
#include "pttt_state.h"
#include "traverser.h"
#include "types.h"

namespace py = pybind11;
using NdArray = py::array_t<Real, py::array::c_style>;

RealBuf to_span(NdArray &arr) {
  return RealBuf(arr.mutable_data(), arr.size());
}

namespace {
constexpr CfrConf default_cfr_args = CfrConf();
std::string infoset_desc(uint64_t key) {
  std::string out = "";
  for (; key; key >>= 5) {
    out += (key & 1) ? '*' : '.';
    out += std::to_string(((key & 0b11110) >> 1) - 1);
  }
  reverse(out.begin(), out.end());
  return out;
}
} // namespace

struct EvExplPy {
  Real ev0;
  std::array<NdArray, 2> gradient;
  std::array<Real, 2> expl;
  std::array<NdArray, 2> best_response;

  // NB: allows implicit conversion
  EvExplPy(const EvExpl &ev) : ev0(ev.ev0), expl(ev.expl) {
    for (int p = 0; p < 2; ++p) {
      gradient[p] = NdArray(std::array<size_t, 2>{ev.gradient[p].size() / 9, 9},
                            &ev.gradient[p][0]);
      best_response[p] =
          NdArray(std::array<size_t, 2>{ev.best_response[p].size() / 9, 9},
                  &ev.best_response[p][0]);
    }
  }
};

template <typename T>
void register_types(py::module &m, const std::string &prefix) {
  py::class_<T>(m, (prefix + "State").c_str())
      .def(py::init())
      .def("clone", [](T &s) -> T { return s; })
      .def("player",
           [](T &s) -> std::optional<uint8_t> {
             if (!s.is_terminal()) {
               return s.player();
             } else {
               return std::nullopt;
             }
           })
      .def("next",
           [](T &s, const uint8_t cell) -> void {
             CHECK(cell < 9, "Invalid cell (must be in range [0..8]; found %d)",
                   cell);
             CHECK(!s.is_terminal(), "Game is over");
             const uint32_t a = s.available_actions();
             CHECK(a & (1 << cell), "The action is not legal");
             s.next(cell);
           })
      .def("is_terminal", &T::is_terminal)
      .def("winner",
           [](const T &s) -> std::optional<uint8_t> {
             CHECK(
                 s.is_terminal(),
                 "Game is not over (you can check with `state.is_terminal()`)");
             const uint8_t w = s.winner();
             if (w < 2) {
               return w;
             } else {
               assert(w == TIE);
               return std::nullopt;
             }
           })
      .def("action_mask",
           [](const T &s) -> std::array<bool, 9> {
             std::array<bool, 9> mask;
             for (uint32_t i = 0, a = s.available_actions(); i < 9;
                  ++i, a >>= 1) {
               mask[i] = a & 1;
             }
             return mask;
           })
      .def("infoset_desc",
           [](const T &s) -> std::string {
             return infoset_desc(s.get_infoset());
           })
      .def("__str__", &T::to_string)
      .def("__repr__", &T::to_string);

  py::class_<Traverser<T>, std::shared_ptr<Traverser<T>>>(
      m, (prefix + "Traverser").c_str())
      .def(py::init<>())
      .def("ev_and_exploitability",
           [](Traverser<T> &traverser, NdArray strat0,
              NdArray strat1) -> EvExplPy {
             // clang-format off
            CHECK(strat0.ndim() == 2 &&
                      strat0.shape(0) == traverser.treeplex[0]->num_infosets() &&
                      strat0.shape(1) == 9,
                  "Invalid shape for Player 1's strategy. Must be (%d, 9); found (%lu, %lu)", 
                  traverser.treeplex[0]->num_infosets(), strat0.shape(0),
                  strat0.shape(1));
            CHECK(strat1.ndim() == 2 &&
                      strat1.shape(0) == traverser.treeplex[1]->num_infosets() &&
                      strat1.shape(1) == 9,
                  "Invalid shape for Player 2's strategy. Must be (%d, 9); (%lu, %lu)",
                  traverser.treeplex[1]->num_infosets(), strat1.shape(0),
                  strat1.shape(1));
             // clang-format on

             return traverser.ev_and_exploitability(
                 {to_span(strat0), to_span(strat1)});
           })
      .def("get_averager", &Traverser<T>::new_averager)
      .def("infoset_desc",
           [](const Traverser<T> &traverser, const uint8_t p,
              const uint32_t row) -> std::string {
             CHECK(p == 0 || p == 1,
                   "Invalid player (expected 0 or 1; found %d)", p);
             CHECK(row < traverser.treeplex[p]->num_infosets(),
                   "Invalid row (expected < %d; found %d)",
                   traverser.treeplex[p]->num_infosets(), row);
             uint64_t key = traverser.treeplex[p]->infoset_keys.at(row);
             return infoset_desc(key);
           })
      .def("construct_uniform_strategies",
           [](const Traverser<T> &traverser) -> std::array<NdArray, 2> {
             std::array<NdArray, 2> out;

             for (int p = 0; p < 2; ++p) {
               const uint32_t rows = traverser.treeplex[p]->num_infosets();
               std::valarray<Real> strategy(0.0, rows * 9);
               traverser.treeplex[p]->set_uniform(strategy);
               out[p] =
                   NdArray(std::array<py::ssize_t, 2>{rows, 9}, &strategy[0]);
             }

             return out;
           })
      .def("parent_index_and_action",
           [](const Traverser<T> &traverser, const uint8_t p,
              const uint32_t row) {
             CHECK(p == 0 || p == 1,
                   "Invalid player (expected 0 or 1; found %d)", p);
             CHECK(row < traverser.treeplex[p]->num_infosets(),
                   "Invalid row (expected < %d; found %d)",
                   traverser.treeplex[p]->num_infosets(), row);
             const auto key = traverser.treeplex[p]->infoset_keys.at(row);
             return std::make_pair(traverser.treeplex[p]->parent_index.at(row),
                                   parent_action(key));
           })
      .def_property(
          "NUM_INFOS_PL1",
          [](const Traverser<T> &traverser) {
            return traverser.treeplex[0]->num_infosets();
          },
          nullptr)
      .def_property(
          "NUM_INFOS_PL2",
          [](const Traverser<T> &traverser) {
            return traverser.treeplex[1]->num_infosets();
          },
          nullptr);

  py::class_<CfrSolver<T>>(m, (prefix + "Solver").c_str())
      .def(py::init())
      .def("step", &CfrSolver<T>::step);

  m.def("CfrSolver",
        [](const std::shared_ptr<Traverser<T>> t, const CfrConf conf)
            -> CfrSolver<T> { return CfrSolver<T>(t, conf); });
}

PYBIND11_MODULE(pydh3, m) {
  py::class_<EvExplPy>(m, "EvExpl")
      .def_readonly("ev0", &EvExplPy::ev0)
      .def_readonly("expl", &EvExplPy::expl)
      .def_readonly("gradient", &EvExplPy::gradient)
      .def_readonly("best_response", &EvExplPy::best_response);

  py::enum_<AveragingStrategy>(m, "AveragingStrategy")
      .value("UNIFORM", AveragingStrategy::UNIFORM)
      .value("LINEAR", AveragingStrategy::LINEAR)
      .value("QUADRATIC", AveragingStrategy::QUADRATIC);

  py::class_<CfrConf>(m, "CfrConf")
      .def(
          py::init(),
          [](AveragingStrategy avg, bool alternation, bool dcfr, bool rmplus,
             bool pcfrp) -> CfrConf {
            return {
                .avg = avg,
                .alternation = alternation,
                .dcfr = dcfr,
                .rmplus = rmplus,
                .pcfrp = pcfrp,
            };
          },
          py::kw_only(), py::arg("avg") = default_cfr_args.avg,
          py::arg("alternation") = default_cfr_args.alternation,
          py::arg("dcfr") = default_cfr_args.dcfr,
          py::arg("rmp") = default_cfr_args.rmplus,
          py::arg("pcfrp") = default_cfr_args.pcfrp)
      .def_readwrite("avg", &CfrConf::avg)
      .def_readwrite("alternation", &CfrConf::alternation)
      .def_readwrite("dcfr", &CfrConf::dcfr)
      .def_readwrite("pcfrp", &CfrConf::pcfrp)
      .def_readwrite("rmplus", &CfrConf::rmplus);

  register_types<DhState<false>>(m, "Dh");
  register_types<DhState<true>>(m, "AbruptDh");
  register_types<PtttState<false>>(m, "Pttt");
  register_types<PtttState<true>>(m, "AbruptPttt");
}