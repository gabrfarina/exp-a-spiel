#include "averager.h"
#include "traverser.h"
#include <cmath>
#include <memory>

namespace {
inline Real iter_weight(const AveragingStrategy avg, const uint32_t iteration) {
  CHECK(iteration > 0, "Iteration must be positive");
  switch (avg) {
  case UNIFORM:
    return 1.0 / iteration;
  case LINEAR:
    return 2.0 / (iteration + 1);
  case QUADRATIC:
    // The weight of iteration t is proportional to t^2. Using the fact that
    // 1^2 + ... + t^2 = t (t + 1) (2t + 1) / 6, we obtain that
    //
    // avg^T = 6 * ((T-1)T(2T-1)/6 * avg^{T-1} + T^2 x^T) / T(T+1)(2T+1)
    //       = ( (T-1)T(2T-1) avg^{T-1} + 6T^2 x^T ) / T(T+1)(2T+1)
    //       = ( (T-1)(2T-1) avg^{T-1} + 6T x^T ) / (T+1)(2T+1)
    //
    // So, comb = 6T / (T+1)(2T+1).
    return 6.0 * iteration / ((iteration + 1.0) * (2 * iteration + 1.0));
  case EXPERIMENTAL:
    return 6.0 * iteration / ((iteration + 1.0) * (iteration + 2.0));
  case LAST:
    return 1.0;
  default:
    CHECK(false, "Unknown averaging strategy");
  }
};
} // namespace

Averager::Averager(std::shared_ptr<Treeplex> treeplex,
                   const AveragingStrategy avg)
    : treeplex_(treeplex), avg_(avg), sf_(0.0, treeplex->num_infosets() * 9),
      buf_(0.0, treeplex->num_infosets() * 9) {}

void Averager::push(ConstRealBuf strategy, const std::optional<Real> weight) {
  if (avg_ == CUSTOM) {
    CHECK(!!weight && *weight >= 0.0, "Weight for custom averaging strategy "
                                      "must be specified and nonnegative");
    if (*weight == 0)
      return; //gracefully handle zero weight
  } else {
    CHECK(!weight, "Cannot specify weight for non-CUSTOM averaging strategies");
  }

  CHECK(strategy.size() == sf_.size(), "Strategy size mismatch");
  CHECK(treeplex_->is_valid_strategy(strategy), "Invalid strategy");

  ++num_;
  Real alpha = 0.;
  if (avg_ != CUSTOM) {
    alpha = iter_weight(avg_, num_);
    CHECK(num_ > 1 || alpha == 1., "The first iteration should have alpha = 1");
  } else {
    weight_sum_ += *weight;
    alpha = *weight / weight_sum_;
  }
  CHECK(std::isfinite(alpha) && alpha <= 1 && alpha >= 0, "Invalid alpha %f", alpha);
  INFO("Pushing strategy with alpha %f", alpha);
  buf_.resize(strategy.size());
  std::copy(strategy.begin(), strategy.end(), std::begin(buf_));
  treeplex_->bh_to_sf(buf_);
  buf_ *= alpha;
  sf_ *= 1.0 - alpha;
  sf_ += buf_;
}

std::valarray<Real> Averager::running_avg() const {
  CHECK(num_ > 0, "No data to average");
  if (avg_ == CUSTOM)
    CHECK(weight_sum_ > 0., "Weight sum is 0");
  std::valarray<Real> out = sf_;
  treeplex_->sf_to_bh(out);
  assert(treeplex_->is_valid_strategy(out));
  return out;
}

void Averager::clear() {
  weight_sum_ = 0.;
  num_ = 0;
  sf_ = 0.;
}

std::string avg_str(const AveragingStrategy avg) {
  switch (avg) {
  case UNIFORM:
    return "uniform";
  case LINEAR:
    return "linear";
  case QUADRATIC:
    return "quadratic";
  case EXPERIMENTAL:
    return "experimental";
  case LAST:
    return "last";
  case CUSTOM:
    return "custom";
  default:
    CHECK(false, "Unknown averaging strategy %d", avg);
  }
}