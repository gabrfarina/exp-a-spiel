#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sys/types.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
#include <valarray>

#include "averager.h"
#include "cfr.h"
#include "dh_state.h"
#include "log.h"
#include "pttt_state.h"
#include "traverser.h"
#include "utils.h"

namespace py = pybind11;
using NdArray = py::array_t<Real, py::array::c_style>;

ConstRealBuf to_const_span(const NdArray &arr) {
  return ConstRealBuf(arr.data(), arr.size());
}

std::array<py::ssize_t, 2> mat_shape(ConstRealBuf buf) {
  CHECK(buf.size() % 9 == 0, "Buffer size must be a multiple of 9");
  return {(py::ssize_t)buf.size() / 9, 9};
}

namespace {
constexpr CfrConf default_cfr_args = CfrConf();

template <size_t N>
auto to_ndarray(std::array<py::ssize_t, N> shape, ConstRealBuf buf) {
  CHECK(prod(shape) == (ssize_t)buf.size(),
        "Shape does not match buffer size (expected %lu; found %lu)",
        buf.size(), prod(shape));
  return NdArray(shape, buf.data());
}
auto to_ndarray(ConstRealBuf buf) { return to_ndarray(mat_shape(buf), buf); }
} // namespace

struct EvExplPy {
  Real ev0;
  std::array<NdArray, 2> gradient;
  std::array<Real, 2> expl;
  std::array<NdArray, 2> best_response;

  // NB: allows implicit conversion
  EvExplPy(const EvExpl &ev) : ev0(ev.ev0), expl(ev.expl) {
    for (auto p : {0, 1}) {
      gradient[p] = to_ndarray(ev.gradient[p]);
      best_response[p] = to_ndarray(ev.best_response[p]);
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
           [](Traverser<T> &traverser, const NdArray &strat0,
              const NdArray &strat1) -> EvExplPy {
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
                 {to_const_span(strat0), to_const_span(strat1)});
           })
      .def(
          "infoset_desc",
          [](const Traverser<T> &traverser, const uint8_t p,
             const uint32_t row) -> std::string {
            CHECK(p == 0 || p == 1,
                  "Invalid player (expected 0 or 1; found %d)", p);
            CHECK(row < traverser.treeplex[p]->num_infosets(),
                  "Invalid row (expected < %d; found %d)",
                  traverser.treeplex[p]->num_infosets(), row);
            uint64_t key = traverser.treeplex[p]->infoset_keys.at(row);
            return infoset_desc(key);
          },
          py::arg("player"), py::arg("row"))
      .def("construct_uniform_strategies",
           [](const Traverser<T> &traverser) -> std::array<NdArray, 2> {
             std::array<NdArray, 2> out;

             for (int p = 0; p < 2; ++p) {
               const uint32_t rows = traverser.treeplex[p]->num_infosets();
               std::valarray<Real> strategy(0.0, rows * 9);
               traverser.treeplex[p]->set_uniform(strategy);
               out[p] = to_ndarray(strategy);
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
          nullptr)
      .def("new_averager", &Traverser<T>::new_averager);

  py::class_<CfrSolver<T>>(m, (prefix + "Solver").c_str())
      .def(py::init([](const std::shared_ptr<Traverser<T>> t, CfrConf conf)
                        -> CfrSolver<T> { return CfrSolver<T>(t, conf); }))
      .def("step", &CfrSolver<T>::step)
      .def("avg_bh", [](const CfrSolver<T> &solver) {
        return std::make_tuple(to_ndarray(solver.get_avg_bh(0)),
                               to_ndarray(solver.get_avg_bh(1)));
      });

  m.def("CfrSolver",
        [](const std::shared_ptr<Traverser<T>> t,
           const CfrConf &conf) -> CfrSolver<T> { return {t, conf}; });
}

PYBIND11_MODULE(pydh3, m) {
  py::class_<EvExplPy>(m, "EvExpl")
      .def_readonly("ev0", &EvExplPy::ev0)
      .def_readonly("expl", &EvExplPy::expl)
      .def_readonly("gradient", &EvExplPy::gradient)
      .def_readonly("best_response", &EvExplPy::best_response)
      .def("__repr__", [](const EvExplPy &ev) {
        std::ostringstream ss;
        ss << std::fixed << std::showpoint << std::setprecision(4)
           << std::setw(5) << "EvExpl(ev0=" << ev.ev0 << ", expl=["
           << ev.expl[0] << ", " << ev.expl[1] << "])";
        return ss.str();
      });

  py::class_<Averager>(m, "Averager")
      .def(
          "push",
          [](Averager &avg, const NdArray &arr, const Real weight) -> void {
            avg.push(to_const_span(arr), weight);
          },
          py::arg("strategy"), py::arg("weight"))
      .def("running_avg",
           [](const Averager &a) { return to_ndarray(a.running_avg()); })
      .def("clear", &Averager::clear);

  py::enum_<AveragingStrategy>(m, "AveragingStrategy")
      .value("UNIFORM", AveragingStrategy::UNIFORM)
      .value("LINEAR", AveragingStrategy::LINEAR)
      .value("QUADRATIC", AveragingStrategy::QUADRATIC);

  py::class_<CfrConf>(m, "CfrConf")
      .def(py::init([](AveragingStrategy avg, bool alternation, bool dcfr,
                       bool rmplus, bool predictive) -> CfrConf {
             return {
                 .avg = avg,
                 .alternation = alternation,
                 .dcfr = dcfr,
                 .rmplus = rmplus,
                 .predictive = predictive,
             };
           }),
           py::kw_only(), py::arg("avg") = default_cfr_args.avg,
           py::arg("alternation") = default_cfr_args.alternation,
           py::arg("dcfr") = default_cfr_args.dcfr,
           py::arg("rmplus") = default_cfr_args.rmplus,
           py::arg("predictive") = default_cfr_args.predictive)
      .def_readwrite("avg", &CfrConf::avg)
      .def_readwrite("alternation", &CfrConf::alternation)
      .def_readwrite("dcfr", &CfrConf::dcfr)
      .def_readwrite("predictive", &CfrConf::predictive)
      .def_readwrite("rmplus", &CfrConf::rmplus)
      .def("__repr__",
           [](const CfrConf &conf) {
             std::ostringstream ss;
             ss << std::boolalpha << "CfrConf(avg=" << avg_str(conf.avg)
                << ", alternation=" << conf.alternation
                << ", dcfr=" << conf.dcfr << ", rmplus=" << conf.rmplus
                << ", predictive=" << conf.predictive << ")";
             return ss.str();
           })
      .def_property_readonly_static("PCFRP",
                                    [](py::handle) -> CfrConf {
                                      return CfrConf{
                                          .avg = AveragingStrategy::QUADRATIC,
                                          .alternation = true,
                                          .dcfr = false,
                                          .rmplus = true,
                                          .predictive = true};
                                    })
      .def_property_readonly_static("DCFR",
                                    [](py::handle) -> CfrConf {
                                      return CfrConf{
                                          .avg = AveragingStrategy::QUADRATIC,
                                          .alternation = true,
                                          .dcfr = true,
                                          .rmplus = false,
                                          .predictive = false};
                                    })
      .def(py::pickle(
          [](const CfrConf &c) {
            return py::make_tuple(c.avg, c.alternation, c.dcfr, c.rmplus,
                                  c.predictive);
          },
          [](py::tuple t) {
            return CfrConf{
                .avg = t[0].cast<AveragingStrategy>(),
                .alternation = t[1].cast<bool>(),
                .dcfr = t[2].cast<bool>(),
                .rmplus = t[3].cast<bool>(),
                .predictive = t[4].cast<bool>(),
            };
          }));

  register_types<DhState<false>>(m, "Dh");
  register_types<DhState<true>>(m, "AbruptDh");
  register_types<CornerDhState>(m, "CornerDh");
  register_types<PtttState<false>>(m, "Pttt");
  register_types<PtttState<true>>(m, "AbruptPttt");
}
