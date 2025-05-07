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

__global__ void ScalarGeKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                               size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] >= val;
  }
}

void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out) {
  ScalarGeKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t *a, scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = log(a[i]);
  }
}

void EwiseLog(const CudaArray &a, CudaArray *out) {
  EwiseLogKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t *a, scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = exp(a[i]);
  }
}

void EwiseExp(const CudaArray &a, CudaArray *out) {
  EwiseExpKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t *a, scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = tanh(a[i]);
  }
}

void EwiseTanh(const CudaArray &a, CudaArray *out) {
  EwiseTanhKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, b.ptr, out->ptr, out->size);
}

__global__ TiledMatMulKernel(scalar_t *a, scalar_t *b, float *out, int m, int n,
                             int p) {
  /*
   * a: m x n
   * b: n x p
   * out: m x p
   */
  __shared__ float ms[TILE][TILE];
  __shared__ float ns[TILE][TILE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE + ty;
  int col = bx * TILE + tx;

  float pvalues = 0.0;

  for (int ph = 0; ph < n / TILE; ph++) {
    ms[ty][tx] = ((row < m) && (ph * TILE + ty) < n)
                     ? a[row * n + ph * TILE + tx]
                     : 0.0f;
    ns[ty][tx] = ((ph * TILE + ty) < n && (col < p))
                     ? b[(ph * TILE * ty) * p + col]
                     : 0.0f;
    __synchthreads();

    for (int i = 0; i < TILE; i++) {
      pvalues += ms[ty][i] * ns[i][tx];
    }
    __syncthreads();
  }

  if ((row < m) && (col < p)) {
    out[row * p + col] = pvalues;
  }
}

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M,
            uint32_t N, uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  TiledMatMulKernel<<<(ceil(M, TILE), ceil(N, TILE), (TILE, TILE)>>>(a.ptr, b.ptr, out.ptr, M, N, P)
}

__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t reduce_size, size_t n) {
  /* Each thread would sum across one reduction item */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t offset = i * reduce_size;
    scalar_t reduce_sum = 0;
    for (int i = 0; i < reduce_size; i++) {
      reduce_sum += a[i + offset];
    }
    out[i] = reduce_sum;
  }
}

void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  ReduceSumKernel<<<ceil(out->size / NUM_THREADS), NUM_THREADS>>>(
      a.ptr, out->ptr, reduce_size, out->size);
}

__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t offset = i * reduce_size;
    scalar_t reduce_max = a[offset];
    for (int i = 1; i < reduce_size; i++) {
      reduce_max = max(reduce_max, a[i + offset]);
    }
    out[i] = reduce_max;
  }
}

void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  ReduceMaxKernel<<<ceil(out->size / NUM_THREADS), NUM_THREADS>>>(
      a.ptr, out->ptr, reduce_size, out->size);
                                           out->size);
}

__device__ threadIndexToIdx(size_t thread_idx, CudaVec shape, CudaVec strides,
                            size_t offset) {
  int dim = shape.size;
  int idx = offset;
  /* Start from last dimension and go backward to get the mapping from the CUDA
   * index (compact array index) to the strided non-compact array index
   */
  for (int i = dim - 1; i >= 0; i--) {
    idx += (thread_idx % shape.data[i]) * strides.data[i];
    thread_idx /= shape.data[i];
  }
  return idx;
}

__global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t n,
                              CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a
   * single entry in the non-compact input a, to the corresponding item (at
   * location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA pointer to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past
   *   passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int idx = threadIndexToIdx(i, shape, stride, offset);
    out[i] = a[idx];
  }
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.
   *
   * Args:
   *   a: non-compact represntation of the array
   *   out: compact version of the array to be written
   *   shape: shape of each dimension for a and out
   *   strides: stride of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   *   compact)
   */

  CompactKernel<<<ceil(out->size / NUM_THREADS), NUM_THREADS>>>(
      a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t n,
                                   CudaVec shape, CudaVec strides,
                                   size_t offset) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[threadIndexToIdx(i, shape, strides, offset)] = a[i];
}

void EwiseSetitem(const CudaArray &a, CudaArray *out,
                  std::vector<int32_t> shape, std::vector<int32_t> strides,
                  size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being
   *   compact)
   */
  EwiseSetitemKernel<<<ceil(out->size, NUM_THREADS), NUM_THREADS>>>(
      a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
}

} // namespace cuda
} // namespace tiny_pytorch
