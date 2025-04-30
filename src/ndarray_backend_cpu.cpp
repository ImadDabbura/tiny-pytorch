#include <cmath>
#include <cstdint>
#include <iostream>

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
   *  void (you need to modify out directly, rather than returning anything;
   * this is true for all the function will implement here, so we won't repeat
   * this note.)
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
} // namespace cpu
} // namespace tiny_pytorch
