#include <cmath>
#include <cstdint>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace tiny_pytorch {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void **)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0)
      throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  scalar_t *ptr;
  size_t size;
};

void Fill(AlignedArray *out, scalar_t val) {
  /* File aligned array with value `val` */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
  /**
   * Set entries in `out` to be the sum of correspondings entires in `a` and
   * `b`.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  /**
   * Set entries in `out` to be the sum of corresponding entry in `a` plus the
   * scalar `val`.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

void EwiseDiv(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void ScalarDiv(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void EwiseMul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void ScalarPower(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

void EwiseMaximum(const AlignedArray &a, const AlignedArray &b,
                  AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseEq(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<scalar_t>(a.ptr[i] == b.ptr[i]);
  }
}

void ScalarEq(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<scalar_t>(a.ptr[i] == val);
  }
}

void EwiseGe(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<scalar_t>(a.ptr[i] >= b.ptr[i]);
  }
}

void ScalarGe(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = static_cast<scalar_t>(a.ptr[i] >= val);
  }
}

void EwiseLog(const AlignedArray &a, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray &a, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray &a, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}

enum strided_mode { INDEX_IN, INDEX_OUT, INDEX_SET };

void _set_noncompact(const AlignedArray *a, AlignedArray *out,
                     std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                     size_t offset, int mode, int val = 0) {
  /*
   * Utility function that would iterate over non_compact array `a` using
   * strided indices and set the elements in `out` array which is compact in
   * sequential order to create compact array
   */
  int cnt = 0;
  int ndim = shape.size();
  std::vector<uint32_t> pos(ndim, 0);
  while (true) {
    int idx = offset;
    for (int i = 0; i < ndim; i++) {
      idx += strides[i] * pos[i];
    }
    switch (mode) {
    case INDEX_IN:
      out->ptr[cnt++] = a->ptr[idx];
      break;
    case INDEX_OUT:
      out->ptr[idx] = a->ptr[cnt++];
      break;
    case INDEX_SET:
      out->ptr[idx] = val;
      break;
    }
    pos[ndim - 1]++;
    int j = ndim - 1;
    while (pos[j] == shape[j]) {
      if (j == 0) {
        return;
      }
      pos[j--] = 0;
      pos[j]++;
    }
  }
}

void Compact(const AlignedArray &a, AlignedArray *out,
             std::vector<uint32_t> shape, std::vector<uint32_t> strides,
             size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   * compact)
   *
   * Returns:
   *  void
   */
  _set_noncompact(&a, out, shape, strides, offset, INDEX_IN);
}

void EwiseSetitem(const AlignedArray &a, AlignedArray *out,
                  std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                  size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being
   * compact)
   */
  _set_noncompact(&a, out, shape, strides, offset, INDEX_OUT);
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out,
                   std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                   size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note
   * be the same as out.size, because out is a non-compact subset array);  it
   * _will_ be the same as the product of items in shape, but convenient to just
   * pass it here. val: scalar value to write to out: non-compact array whose
   * items are to be written shape: shapes of each dimension of out strides:
   * strides of the out array offset: offset of the out array
   */

  _set_noncompact(nullptr, out, shape, strides, offset, INDEX_SET, val);
}

void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  scalar_t sum;
  for (int i = 0; i < out->size; i++) {
    sum = 0;
    for (int j = 0; j < reduce_size; j++) {
      sum += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum;
  }
}

void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size) {
  /**
   * Reduce by taking max over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  scalar_t max;
  for (int i = 0; i < out->size; i++) {
    max = a.ptr[i * reduce_size];
    for (int j = 1; j < reduce_size; j++) {
      max = std::max(max, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = max;
  }
}

void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out,
            uint32_t m, uint32_t n, uint32_t p) {
  /**
   * Naively multiply two (compact) matrices into an output (also compact)
   * matrix.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  scalar_t res;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      res = 0;
      for (int k = 0; k < n; k++) {
        res += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
      out->ptr[i * p + j] = res;
    }
  }
}

inline void AlignedDot(const float *__restrict__ a, const float *__restrict__ b,
                       float *__restrict__ out) {
  /**
   * Multiply together two TILE x TILE matrices, and _add_ the result to out.
   * `__restrict__ keyword indicates to the compiler that a, b, and out don't
   * have any overlapping memory (which is necessary in order for vector
   * operations to be equivalent to their non-vectorized ops.
   * `__builtin_assume_aligned` keyword tells the compiler that the input array
   * will be aligned to the appropriate blocks in memory, which also helps the
   * compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float *)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float *)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float *)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  scalar_t res;
  for (int i = 0; i < TILE; i++) {
    for (int j = 0; j < TILE; j++) {
      res = out[i * TILE + j];
      for (int k = 0; k < TILE; k++) {
        res += a[i * TILE + k] * b[k * TILE + j];
      }
      out[i * TILE + j] = res;
    }
  }
}

void MatmulTiled(const AlignedArray &a, const AlignedArray &b,
                 AlignedArray *out, uint32_t m, uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting,
   * a, b, and out are all *4D* compact arrays of the appropriate size, e.g. a
   * is an array of size a[m/TILE][n/TILE][TILE][TILE].
   *
   * Note that this function will only be called when m, n, p are all multiples
   * of TILE.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  // initialize out to zero
  for (int i = 0; i < m * p; i++)
    out->ptr[i] = 0;

  for (int i = 0; i < m / TILE; i++) {
    for (int j = 0; j < p / TILE; j++) {
      for (int k = 0; k < n / TILE; k++) {
        AlignedDot(&a.ptr[i * n * TILE + k * TILE * TILE],
                   &b.ptr[k * p * TILE + j * TILE * TILE],
                   &out->ptr[i * p * TILE + j * TILE * TILE]);
      }
    }
  }
}

} // namespace cpu
} // namespace tiny_pytorch

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace tiny_pytorch;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // TODO: Creates a view now, we should change it to create a copy
  m.def("to_numpy", [](const AlignedArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t &c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray *out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });
  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
