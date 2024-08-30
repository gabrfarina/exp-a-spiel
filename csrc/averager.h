#pragma once

#include "log.h"
#include "types.h"
#include <memory>
#include <valarray>

enum AveragingStrategy { UNIFORM, LINEAR, QUADRATIC };

struct Treeplex;

inline Real iter_weight(const AveragingStrategy avg, const uint32_t iteration) {
  CHECK(iteration > 0, "Iteration must be positive");
  switch (avg) {
  case UNIFORM:
    return 1.0;
  case LINEAR:
    return iteration;
  case QUADRATIC:
    return 1.0 * iteration * iteration;
  default:
    CHECK(false, "Unknown averaging strategy");
  }
}

class Averager {
public:
  Averager(std::shared_ptr<Treeplex> treeplex);

  void push(ConstRealBuf strategy, const Real weight);
  std::valarray<Real> running_avg() const;
  void clear() {
    sum_weights_ = 0.0;
    sf_ = 0.0;
  }
private:
  std::shared_ptr<Treeplex> treeplex_;
  Real sum_weights_ = 0.0;
  std::valarray<Real> sf_;
  std::valarray<Real> buf_;
};