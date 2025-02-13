#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <sys/types.h>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "dh_state.h"
#include "log.h"
#include "pttt_state.h"
#include "traverser.h"

int main() {
  INFO("starting eas (num threads: %d)", omp_get_max_threads());
#ifdef DEBUG
  INFO("DEBUG is defined");
#endif

  Traverser<DhState<true>> traverser;
  std::valarray<Real> strategies[2];

  strategies[0].resize(traverser.treeplex[0]->num_infosets() * 9, 0.0);
  strategies[1].resize(traverser.treeplex[1]->num_infosets() * 9, 0.0);

  traverser.treeplex[0]->set_uniform(strategies[0]);
  traverser.treeplex[1]->set_uniform(strategies[1]);

  traverser.ev_and_exploitability({strategies[0], strategies[1]});
}
