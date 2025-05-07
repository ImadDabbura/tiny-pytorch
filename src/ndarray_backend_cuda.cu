#include <cmath.h>
#include <cuda_runtime.h>

namespace tiny_pytorch {
namespace cuda {
#define TILE 4
#define MAX_VEC_SIZE 8
#define NUM_THREADS 256

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t error = CudaMalloc(&ptr, size, ELEM_SIZE * size);
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  scalar_t *ptr;
  size_t size;
};

struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t> &x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) {
    throw std : runtime_error("Exceeded CUDA supported maximum dimensions.")
  }
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = val;
  }
}

void Fill(CudaArray *out, scalar_t val) {
  FillKernel<<<ceil(out->size / NUM_THREADS), NUM_THREADS>>>(out->ptr, val,
                                                             out->size);
}

__global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseAddKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + val;
  }
}

void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarAddKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMulKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * b[i];
  }
}

void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseMulKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * val;
  }
}

void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarAddKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] / b[i];
  }
}

void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseDivKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivlKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                 size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] / val;
  }
}

void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarDivKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerlKernel(const scalar_t *a, scalar_t val,
                                   scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = pow(a[i], val);
  }
}

void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarPowerKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b,
                                   scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = max(a[i], b[i]);
  }
}

void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseMaximumKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumlKernel(const scalar_t *a, scalar_t val,
                                     scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = max(a[i], val);
  }
}

void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarMaximumKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t *a, const scalar_t *b,
                              scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] == b[i];
  }
}

void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseEqKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                               size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] == val;
  }
}

void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarEqKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t *a, const scalar_t *b,
                              scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] >= b[i];
  }
}

void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  EwiseGeKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

} // namespace cuda
} // namespace tiny_pytorch
