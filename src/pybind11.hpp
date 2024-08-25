#ifndef PYBIND11_HPP
#define PYBIND11_HPP

// We must load pybind11 consistently through this header to ensure that our type_caster
// specialization is used in every translation unit. Otherwise we get ODR violations.

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <vector>

namespace pybind11 {
namespace detail {

// This specialization for vector<double> acts as an override from the more generic
// template in pybind11/stl.h. We use it to cast the C++ vector into a numpy array
// instead of a Python list.
template <>
struct type_caster<std::vector<double>> {
  using vec_t = std::vector<double>;
  using value_conv = make_caster<double>;
  using size_conv = make_caster<std::size_t>;

  bool load(handle src, bool convert) {
    value.clear();
    // TODO optimize for python objects that support buffer protocol
    if (isinstance<iterable>(src)) {
      auto seq = reinterpret_borrow<iterable>(src);
      if (hasattr(seq, "__len__")) value.reserve(static_cast<std::size_t>(len(seq)));
      for (auto it : seq) {
        value_conv conv;
        if (!conv.load(it, convert)) return false;
        value.push_back(cast_op<double&&>(std::move(conv)));
      }
      return true;
    }
    return false;
  }

  template <typename T>
  static handle cast(T&& src, return_value_policy, handle) {
    array_t<double> arr(static_cast<ssize_t>(src.size()));
    std::copy(src.begin(), src.end(), arr.mutable_data());
    return arr.release();
  }

  PYBIND11_TYPE_CASTER(vec_t, _("List[") + value_conv::name + _("]"));
};

} // namespace detail
} // namespace pybind11

#endif // PYBIND11_HPP
