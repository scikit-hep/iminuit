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
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!std::is_lvalue_reference<T>::value)
      policy = return_value_policy_override<Value>::policy(policy);
    // TODO optimize by returning ndarray instead of list
    list l(src.size());
    size_t index = 0;
    for (auto&& value : src) {
      auto value_ = reinterpret_steal<object>(value_conv::cast(value, policy, parent));
      if (!value_) return handle();
      PyList_SET_ITEM(l.ptr(), (ssize_t)index++,
                      value_.release().ptr()); // steals a reference
    }
    return l.release();
  }

  PYBIND11_TYPE_CASTER(vec_t, _("List[") + value_conv::name + _("]"));
};

} // namespace detail
} // namespace pybind11
