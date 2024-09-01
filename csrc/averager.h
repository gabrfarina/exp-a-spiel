#pragma once

#include <memory>
#include <valarray>

#include "log.h"
#include "utils.h"

enum AveragingStrategy { UNIFORM, LINEAR, QUADRATIC, EXPERIMENTAL, LAST };
std::string avg_str(const AveragingStrategy avg);

struct Treeplex;

class Averager {
public:
  Averager(std::shared_ptr<Treeplex> treeplex,
           const AveragingStrategy avg = UNIFORM);

  void push(ConstRealBuf strategy);
  std::valarray<Real> running_avg() const;
  void clear() {
    num_ = 0;
    sf_ = 0.;
  }

private:
  std::shared_ptr<Treeplex> treeplex_;
  AveragingStrategy avg_;
  size_t num_ = 0;

  std::valarray<Real> sf_;
  std::valarray<Real> buf_;
};