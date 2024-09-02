#pragma once

#include <memory>
#include <optional>
#include <valarray>

#include "log.h"
#include "utils.h"

enum AveragingStrategy {
  UNIFORM,
  LINEAR,
  QUADRATIC,
  EXPERIMENTAL,
  LAST,
  CUSTOM
};
std::string avg_str(const AveragingStrategy avg);

struct Treeplex;

class Averager {
public:
  Averager(std::shared_ptr<Treeplex> treeplex,
           const AveragingStrategy avg = UNIFORM);

  // Weight must be set for CUSTOM averaging strategy, and must be none for
  // other averaging strategies.
  void push(ConstRealBuf strategy,
            const std::optional<Real> weight = std::nullopt);
  std::valarray<Real> running_avg() const;
  void clear();

private:
  std::shared_ptr<Treeplex> treeplex_;
  AveragingStrategy avg_;
  Real weight_sum_ = 0.;
  size_t num_ = 0;

  std::valarray<Real> sf_;
  std::valarray<Real> buf_;
};