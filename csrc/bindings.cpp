#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/types.h>
#include <valarray>

#include "log.h"
#include "state.h"
#include "traverser.h"

namespace py = pybind11;

PYBIND11_MODULE(pydh3, m) {
  m.attr("NUM_INFOS_PL1") = py::int_(NUM_INFOS_PL1);
  m.attr("NUM_INFOS_PL2") = py::int_(NUM_INFOS_PL2);

  py::class_<DhState>(m, "State")
      .def(py::init(&DhState::root))
      .def("player",
           [](DhState &s) -> std::optional<uint8_t> {
             if (s.winner() == 0xff) {
               return s.player();
             } else {
               return std::nullopt;
             }
           })
      .def("next",
           [](DhState &s, const uint8_t cell) -> void {
             CHECK(cell < 9, "Invalid cell (must be in range [0..8]; found %d)",
                   cell);
             CHECK(s.winner() == 0xff, "Game is over");
             const uint32_t a = s.available_actions();
             CHECK(a & (1 << cell), "The action is not legal");
             s.next(cell);
           })
      .def("next_abrupt",
           [](DhState &s, const uint8_t cell) -> void {
             CHECK(cell < 9, "Invalid cell (must be in range [0..8]; found %d)",
                   cell);
             CHECK(s.winner() == 0xff, "Game is over");
             const uint32_t a = s.available_actions();
             CHECK(a & (1 << cell), "The action is not legal");
             s.next_abrupt(cell);
           })
      .def("winner",
           [](const DhState &s) -> std::optional<uint8_t> {
             const uint8_t w = s.winner();
             if (w != 0xff) {
               return w;
             } else {
               return std::nullopt;
             }
           })
      .def("action_mask",
           [](const DhState &s) -> std::array<bool, 9> {
             std::array<bool, 9> mask;
             for (uint32_t i = 0, a = s.available_actions(); i < 9;
                  ++i, a >>= 1) {
               mask[i] = a & 1;
             }
             return mask;
           })
      .def("infoset", &DhState::get_infoset)
      .def("__str__", &DhState::debug_string)
      .def("__repr__", &DhState::debug_string);

  py::class_<EvExpl>(m, "EvExpl")
      .def_readonly("ev0", &EvExpl::ev0)
      .def_readonly("expl", &EvExpl::expl);

  py::class_<DhTraverser>(m, "Traverser")
      .def(py::init<>())
      .def(
          "ev_and_exploitability",
          [](DhTraverser &traverser,
             py::array_t<Real, py::array::c_style> strat0,
             py::array_t<Real, py::array::c_style> strat1) -> EvExpl {
            CHECK(
                strat0.ndim() == 2 && strat0.shape(0) == NUM_INFOS_PL1 &&
                    strat0.shape(1) == 9,
                "Invalid shape for Player 1's strategy. Must be (%d, 9); found "
                "(%lu, %lu)",
                NUM_INFOS_PL1, strat0.shape(0), strat0.shape(1));
            CHECK(
                strat1.ndim() == 2 && strat1.shape(0) == NUM_INFOS_PL2 &&
                    strat1.shape(1) == 9,
                "Invalid shape for Player 2's strategy. Must be (%d, 9); found "
                "(%lu, %lu)",
                NUM_INFOS_PL2, strat1.shape(0), strat1.shape(1));

            return traverser.ev_and_exploitability(
                {strat0.data(), strat1.data()});
          })
      .def("infoset_desc",
           [](const DhTraverser &traverser, const uint8_t p,
              const uint32_t row) -> std::string {
             CHECK(p == 0 || p == 1,
                   "Invalid player (expected 0 or 1; found %d)", p);
             CHECK(row < traverser.treeplex[p].num_infosets(),
                   "Invalid row (expected < %d; found %d)",
                   traverser.treeplex[p].num_infosets(), row);
             uint64_t key = traverser.treeplex[p].infoset_keys.at(row);
             std::string out = "";
             for (; key; key >>= 5) {
               out += (key & 1) ? '*' : '.';
               out += std::to_string(((key & 0b11110) >> 1) - 1);
             }
             reverse(out.begin(), out.end());
             return out;
           })
      .def("construct_uniform_strategies",
           [](const DhTraverser &traverser)
               -> std::array<py::array_t<Real, py::array::c_style>, 2> {
             std::array<py::array_t<Real, py::array::c_style>, 2> out;

             for (int p = 0; p < 2; ++p) {
               const uint32_t rows = traverser.treeplex[p].num_infosets();
               std::valarray<Real> strategy(0.0, rows * 9);
               traverser.treeplex[p].set_uniform(&strategy[0]);
               out[p] = py::array_t<Real, py::array::c_style>(
                   std::array<py::ssize_t, 2>{rows, 9}, &strategy[0]);
             }

             return out;
           });
}