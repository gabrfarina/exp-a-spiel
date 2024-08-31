#include "averager.h"
#include "traverser.h"

Averager::Averager(std::shared_ptr<Treeplex> treeplex)
    : treeplex_(treeplex), sum_weights_(0.0),
      sf_(0.0, treeplex->num_infosets() * 9),
      buf_(0.0, treeplex->num_infosets() * 9) {}

void Averager::push(ConstRealBuf strategy, const Real weight) {
  CHECK(strategy.size() == sf_.size(), "Strategy size mismatch");
  CHECK(weight >= 0.0, "Averaging weight must be nonnegative (found: %f)",
        weight);

  treeplex_->validate_strategy(strategy);

  sum_weights_ += weight;
  auto alpha = weight / sum_weights_;
  INFO("Pushing strategy with weight %f and alpha %f", weight, alpha);
  buf_.resize(strategy.size());
  std::copy(strategy.begin(), strategy.end(), std::begin(buf_));
  treeplex_->bh_to_sf(buf_);
  buf_ *= alpha;
  sf_ *= 1.0 - alpha;
  sf_ += buf_;
}

std::valarray<Real> Averager::running_avg() const {
  CHECK(sum_weights_ > 0.0, "No data to average");
  std::valarray<Real> out = sf_;
  treeplex_->sf_to_bh(out);
  treeplex_->validate_strategy(out);
  return out;
}

std::string avg_str(const AveragingStrategy avg) {
  switch (avg) {
  case UNIFORM:
    return "uniform";
  case LINEAR:
    return "linear";
  case QUADRATIC:
    return "quadratic";
  default:
    CHECK(false, "Unknown averaging strategy %d", avg);
  }
}