#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace pybind11 {
namespace detail {

template <typename Value>
struct type_caster<std::vector<Value>> {
  using vec_t = std::vector<Value>;
  using value_conv = make_caster<Value>;
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
        value.push_back(cast_op<Value&&>(std::move(conv)));
      }
      return true;
    }
    return false;
  }

public:
  template <typename T>
  static handle cast(T&& src, return_value_policy, handle) {
    array_t<Value> arr({static_cast<ssize_t>(src.size())});
    std::copy(src.begin(), src.end(), arr.mutable_data());
    return arr.release();
  }

  PYBIND11_TYPE_CASTER(vec_t, _("List[") + value_conv::name + _("]"));
};

} // namespace detail
} // namespace pybind11
