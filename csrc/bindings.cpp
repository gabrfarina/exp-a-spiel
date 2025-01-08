#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

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
#include "vec_env.h"

namespace py = pybind11;
using NdArray = py::array_t<Real, py::array::c_style>;
using BoolNdArray = py::array_t<bool, py::array::c_style>;
template <typename T = ConstRealBuf> T to_const_span(const NdArray &arr) {
  return T(arr.data(), arr.size());
}

template <typename T = RealBuf> T to_mut_span(NdArray &arr) {
  return T(arr.mutable_data(), arr.size());
}

std::array<py::ssize_t, 2> mat_shape(ConstRealBuf buf) {
  CHECK(buf.size() % 9 == 0, "Buffer size must be a multiple of 9");
  return {(py::ssize_t)buf.size() / 9, 9};
}

namespace {
constexpr CfrConf CFR_DEFAULTS{};

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
  std::tuple<NdArray, NdArray> gradient;
  std::tuple<Real, Real> expl;
  std::tuple<NdArray, NdArray> best_response;

  // NB: allows implicit conversion
  EvExplPy(EvExpl ev) : ev0(ev.ev0), expl{ev.expl[0], ev.expl[1]} {
    std::get<0>(gradient) = to_ndarray(ev.gradient[0]);
    std::get<0>(best_response) = to_ndarray(ev.best_response[0]);
    std::get<1>(gradient) = to_ndarray(ev.gradient[1]);
    std::get<1>(best_response) = to_ndarray(ev.best_response[1]);
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
           }, "Returns the current player, or None if the game is over")
      .def(
          "next",
          [](T &s, const uint8_t cell) -> void {
            CHECK(cell < 9, "Invalid cell (must be in range [0..8]; found %d)",
                  cell);
            CHECK(!s.is_terminal(), "Game is over");
            const uint32_t a = s.available_actions();
            CHECK(a & (1 << cell), "The action is not legal");
            s.next(cell);
          },
          py::arg("cell"), "Play the given cell")
      .def("is_terminal", &T::is_terminal, "Returns True if the game is over")
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
           }, "Returns the winner (0, 1, or None if it's a tie)")
      .def("action_mask",
           [](const T &s) -> std::array<bool, 9> {
             std::array<bool, 9> mask;
             for (uint32_t i = 0, a = s.available_actions(); i < 9;
                  ++i, a >>= 1) {
               mask[i] = a & 1;
             }
             return mask;
           }, "Returns a mask of legal actions")
      .def("infoset_desc",
           [](const T &s) -> std::string {
             return infoset_desc(s.get_infoset());
           }, "Returns a description of the current infoset")
      .def("compute_openspiel_infostate",
           [](const T &s) -> BoolNdArray {
             std::array<bool, T::OPENSPIEL_INFOSTATE_SIZE> buf;
             T::compute_openspiel_infostate(s.player(), s.get_infoset(), buf);
             return BoolNdArray(
                 std::array<py::ssize_t, 1>{T::OPENSPIEL_INFOSTATE_SIZE},
                 buf.data());
           }, "Returns the OpenSpiel infoset")
      .def("__str__", &T::to_string)
      .def("__repr__", &T::to_string)
      .def_property_readonly_static("OPENSPIEL_INFOSTATE_SIZE", [](py::object) {
        return T::OPENSPIEL_INFOSTATE_SIZE;
      }, "The size of the OpenSpiel infoset");

    py::class_<VecEnv<T>>(
            m, (prefix + "VecEnv").c_str()
    )
            .def(py::init<int>())
            .def("reset", [](VecEnv<T> &s, [[maybe_unused]] int seed = 114514) {
                return s.reset();
            })
            .def("step", &VecEnv<T>::step)
            .def("close", &VecEnv<T>::close)
            .def_property_readonly("obs_shape", [](const VecEnv<T> & /*env*/) {
                return std::vector<int>{T::OPENSPIEL_INFOSTATE_SIZE};
            })
            .def_property_readonly("num_envs", [](const VecEnv<T> & envs) {
                return envs.num_envs;
            })
            .def_property_readonly("num_actions", [](const VecEnv<T> & /*env*/) {
                return 9;
            })
            .def_property_readonly("num_players", [](const VecEnv<T> & /*env*/) {
                return 2;
            });



  py::class_<Traverser<T>, std::shared_ptr<Traverser<T>>>(
      m, (prefix + "Traverser").c_str())
      .def(py::init<>())
      .def(
          "ev_and_exploitability",
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
          },
          py::arg("strat0"), py::arg("strat1"))
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
      .def(
          "row_for_infoset",
          [](const Traverser<T> &traverser, const uint8_t p,
             const std::string infoset_desc) -> uint32_t {
            uint64_t infoset_key = 0;
            CHECK(infoset_desc.size() % 2 == 0,
                  "Infoset desc does not have even length");
            for (size_t i = 0; i < infoset_desc.size() / 2; ++i) {
              const char cell = infoset_desc[2 * i];
              const char outcome = infoset_desc[2 * i + 1];
              CHECK(cell >= '0' && cell <= '9',
                    "Invalid cell in infoset desc `%s`", infoset_desc.c_str());
              CHECK(outcome == '*' || outcome == '.',
                    "Invalid outcome in infoset desc `%s`",
                    infoset_desc.c_str());
              infoset_key <<= 5;
              infoset_key += 2 * ((cell - '0') + 1);
              infoset_key += (outcome == '*');
            }

            auto it = traverser.treeplex[p]->infosets.find(infoset_key);
            CHECK(it != traverser.treeplex[p]->infosets.end(),
                  "The given infoset_desc does not exist");
            return it->second.infoset_id;
          },
          py::arg("player"), py::arg("infoset_desc"))
      .def("construct_uniform_strategies",
           [](const Traverser<T> &traverser) -> std::tuple<NdArray, NdArray> {
             PerPlayer<NdArray> out;

             for (int p = 0; p < 2; ++p) {
               const uint32_t rows = traverser.treeplex[p]->num_infosets();
               std::valarray<Real> strategy(0.0, rows * 9);
               traverser.treeplex[p]->set_uniform(strategy);
               out[p] = to_ndarray(strategy);
             }

             return std::make_tuple(out[0], out[1]);
           }, "Constructs uniform (behavioral) strategies for both players")
      .def(
          "compute_openspiel_infostates",
          [](const Traverser<T> &traverser, const uint8_t p) -> BoolNdArray {
            CHECK(p == 0 || p == 1,
                  "Invalid player (expected 0 or 1; found %u)", p);
            const uint32_t nrows = traverser.treeplex[p]->num_infosets();
            std::valarray<bool> buf(nrows * T::OPENSPIEL_INFOSTATE_SIZE);
            traverser.compute_openspiel_infostates(p, buf);
            return BoolNdArray(
                std::array<py::ssize_t, 2>{nrows, T::OPENSPIEL_INFOSTATE_SIZE},
                &buf[0]);
          },
          py::arg("player"))
      .def("compute_openspiel_infostate",
           [](const Traverser<T> &traverser, const uint8_t p,
              const uint32_t state) -> BoolNdArray {
             CHECK(p == 0 || p == 1,
                   "Invalid player (expected 0 or 1; found %u)", p);
             std::array<bool, T::OPENSPIEL_INFOSTATE_SIZE> buf;
             traverser.compute_openspiel_infostate(p, state, buf);
             return BoolNdArray(
                 std::array<py::ssize_t, 1>{T::OPENSPIEL_INFOSTATE_SIZE},
                 buf.data());
           })
      .def("is_valid_strategy",
           [](const Traverser<T> &traverser, const uint8_t p,
              const NdArray &strategy) {
             return traverser.treeplex[p]->is_valid_strategy(
                 to_const_span(strategy));
           }, "Checks if the given strategy is valid, (has the right shape, is nonnegative and sums to 1)")
      .def_property_readonly("NUM_INFOS_PL0",
                             [](const Traverser<T> &traverser) {
                               return traverser.treeplex[0]->num_infosets();
                             }, "Number of infosets for player 0")
      .def_property_readonly("NUM_INFOS_PL1",
                             [](const Traverser<T> &traverser) {
                               return traverser.treeplex[1]->num_infosets();
                             }, "Number of infosets for player 1")
      .def_property_readonly_static(
          "OPENSPIEL_INFOSTATE_SIZE",
          [](py::object) { return T::OPENSPIEL_INFOSTATE_SIZE; }, 
          "The size of the OpenSpiel infoset")
      .def(
          "parent_index_and_action",
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
          },
          py::arg("player"), py::arg("row"))
      .def("new_averager", &Traverser<T>::new_averager, py::arg("player"),
           py::arg("avg_strategy"), "Returns a new Averager object");

  py::class_<CfrSolver<T>>(m, (prefix + "CfrSolver").c_str())
      .def(py::init([](const std::shared_ptr<Traverser<T>> t, CfrConf conf)
                        -> CfrSolver<T> { return CfrSolver<T>(t, conf); }),
           py::arg("traverser"), py::arg("cfr_conf"))
      .def("step", &CfrSolver<T>::step, "Performs a single CFR step")
      .def("avg_bh",
           [](const CfrSolver<T> &solver) -> std::tuple<NdArray, NdArray> {
             return std::make_tuple(to_ndarray(solver.get_avg_bh(0)),
                                    to_ndarray(solver.get_avg_bh(1)));
           }, "Returns the average behavioral strategies");

  m.def(
      "CfrSolver",
      [](const std::shared_ptr<Traverser<T>> t,
         const CfrConf &conf) -> CfrSolver<T> {
        return {t, conf};
      },
      py::arg("traverser"), py::arg("cfr_conf"), "Constructs a new CfrSolver");
}

PYBIND11_MODULE(pydh3, m) {
  py::class_<EvExplPy>(m, "EvExpl")
      .def_readonly("ev0", &EvExplPy::ev0, "EV0")
      .def_readonly("expl", &EvExplPy::expl, "Exploitabilities")
      .def_readonly("gradient", &EvExplPy::gradient, "Gradients")
      .def_readonly("best_response", &EvExplPy::best_response, "Best response policies")
      .def("__repr__", [](const EvExplPy &ev) {
        std::ostringstream ss;
        ss << std::fixed << std::showpoint << std::setprecision(8)
           << "EvExpl(ev0=" << ev.ev0 << ", expl=[" << std::get<0>(ev.expl)
           << ", " << std::get<1>(ev.expl) << "])";
        return ss.str();
      }).doc() = "Utility class for holding EV and exploitability results";

  py::class_<Averager>(m, "Averager")
      .def(
          "push",
          [](Averager &avg, const NdArray &strat,
             const std::optional<Real> weight) -> void {
            avg.push(to_const_span(strat), weight);
          },
          py::arg("strategy"), py::arg("weight") = std::nullopt, "Pushes a strategy, use weight argument for custom averaging")
      .def("running_avg",
           [](const Averager &a) { return to_ndarray(a.running_avg()); })
      .def("clear", &Averager::clear, "Clears the running average")
      .doc() = "Utility class for averaging strategies";

  py::enum_<AveragingStrategy>(m, "AveragingStrategy")
      .value("UNIFORM", AveragingStrategy::UNIFORM, "All strategies have the same weight.")
      .value("LINEAR", AveragingStrategy::LINEAR, "Strategy t has weight proportional to t")
      .value("QUADRATIC", AveragingStrategy::QUADRATIC, "Strategy t has weight proportional to $t^2$")
      .value("EXPERIMENTAL", AveragingStrategy::EXPERIMENTAL, "Experimental averating")
      .value("LAST", AveragingStrategy::LAST, "The last strategy has weight 1")
      .value("CUSTOM", AveragingStrategy::CUSTOM, "The user will provide the weights via the `weigh` argument");

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
           py::kw_only(), py::arg("avg") = CFR_DEFAULTS.avg,
           py::arg("alternation") = CFR_DEFAULTS.alternation,
           py::arg("dcfr") = CFR_DEFAULTS.dcfr,
           py::arg("rmplus") = CFR_DEFAULTS.rmplus,
           py::arg("predictive") = CFR_DEFAULTS.predictive)
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
      .def_property_readonly_static( //
          "PCFRP",
          [](py::handle) -> CfrConf {
            return CfrConf{.avg = AveragingStrategy::QUADRATIC,
                           .alternation = true,
                           .dcfr = false,
                           .rmplus = true,
                           .predictive = true};
          })
      .def_property_readonly_static( //
          "DCFR",
          [](py::handle) -> CfrConf {
            return CfrConf{.avg = AveragingStrategy::QUADRATIC,
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
          }))
      .doc() = R"SDF(
        Configuration for CFR
        - avg: Averaging strategy,
        - alternation: Use alternation,
        - dcfr: Use discounted CFR,
        - predictive: Use predictive CFR
        - rmplus: Use RM+, can be mixed with DCFR, 

        The following configurations are available:
        - PCFRP: Predictive CFR+,
        - DCFR: Discounted CFR,
)SDF";

  register_types<DhState<false>>(m, "Dh");
  register_types<DhState<true>>(m, "AbruptDh");
  register_types<CornerDhState>(m, "CornerDh");
  register_types<PtttState<false>>(m, "Pttt");
  register_types<PtttState<true>>(m, "AbruptPttt");
}
