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

} // namespace cpu
} // namespace tiny_pytorch
