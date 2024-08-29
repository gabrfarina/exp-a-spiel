#pragma once

#include <array>
#include <cstdint>
#include <span>

using Real = double;
using RealBuf = std::span<Real>;
using ConstRealBuf = std::span<const Real>;
template <typename T> using PerPlayer = std::array<T, 2>;
