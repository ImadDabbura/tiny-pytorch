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
    out->ptr[i] = scalar_t(a.ptr[i] == b.ptr[i]);
  }
}

void ScalarEq(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = scalar_t(a.ptr[i] == val);
  }
}

void EwiseGe(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = scalar_t(a.ptr[i] >= b.ptr[i]);
  }
}

void ScalarGe(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = scalar_t(a.ptr[i] >= val);
  }
}

} // namespace cpu
} // namespace tiny_pytorch
