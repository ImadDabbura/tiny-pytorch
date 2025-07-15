#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tiny_pytorch {
namespace cuda {
#define TILE 4
#define MAX_VEC_SIZE 8
#define NUM_THREADS 256

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

// Error checking macro
#define CUDA_CHECK_ERROR(err, op) \
  do { \
    cudaError_t _err = (err); \
    if (_err != cudaSuccess) { \
      throw std::runtime_error(std::string(op) + " failed at " + __FILE__ + ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(_err)); \
    } \
  } while (0)

// Kernel launch utility
template<typename... Args>
void launch_kernel(void (*kernel)(Args...), size_t num_elements, Args... args) {
  dim3 block_size(NUM_THREADS);
  dim3 grid_size((num_elements + NUM_THREADS - 1) / NUM_THREADS);

  kernel<<<grid_size, block_size>>>(args...);

  // Check for kernel launch errors
  cudaError_t error = cudaGetLastError();
  CUDA_CHECK_ERROR(error, "Kernel launch");

  // Synchronize and check for runtime errors
  error = cudaDeviceSynchronize();
  CUDA_CHECK_ERROR(error, "Kernel execution");
}

// 2D kernel launch utility for matrix operations
template<typename... Args>
void launch_kernel_2d(void (*kernel)(Args...), int grid_x, int grid_y, int block_x, int block_y, Args... args) {
  dim3 block_size(block_x, block_y);
  dim3 grid_size(grid_x, grid_y);

  kernel<<<grid_size, block_size>>>(args...);

  // Check for kernel launch errors
  cudaError_t error = cudaGetLastError();
  CUDA_CHECK_ERROR(error, "2D kernel launch");

  // Synchronize and check for runtime errors
  error = cudaDeviceSynchronize();
  CUDA_CHECK_ERROR(error, "2D kernel execution");
}

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t error = cudaMalloc(&ptr, size * ELEM_SIZE);
    CUDA_CHECK_ERROR(error, "cudaMalloc");
    this->size = size;
  }
  ~CudaArray() {
    if (ptr) {
      cudaFree(ptr);
    }
  }
  size_t ptr_as_int() { return (size_t)ptr; }
  scalar_t *ptr;
  size_t size;
};

struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t> &x) {
  CudaVec vec;
  if (x.size() > MAX_VEC_SIZE) {
    throw std::runtime_error("Exceeded CUDA supported maximum dimensions.");
  }
  vec.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    vec.data[i] = x[i];
  }
  return vec;
}

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = val;
  }
}

void Fill(CudaArray *out, scalar_t val) {
  launch_kernel(FillKernel, out->size, out->ptr, val, out->size);
}

__global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  launch_kernel(EwiseAddKernel, out->size, a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + val;
  }
}

void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out) {
  launch_kernel(ScalarAddKernel, out->size, a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMulKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * b[i];
  }
}

void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  launch_kernel(EwiseMulKernel, out->size, a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * val;
  }
}

void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out) {
  launch_kernel(ScalarMulKernel, out->size, a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] / b[i];
  }
}

void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  launch_kernel(EwiseDivKernel, out->size, a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                 size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] / val;
  }
}

void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out) {
  launch_kernel(ScalarDivKernel, out->size, a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t *a, scalar_t val,
                                   scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = pow(a[i], val);
  }
}

void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out) {
  launch_kernel(ScalarPowerKernel, out->size, a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b,
                                   scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = max(a[i], b[i]);
  }
}

void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  launch_kernel(EwiseMaximumKernel, out->size, a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t *a, scalar_t val,
                                     scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = max(a[i], val);
  }
}

void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out) {
  launch_kernel(ScalarMaximumKernel, out->size, a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t *a, const scalar_t *b,
                              scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] == b[i];
  }
}

void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  launch_kernel(EwiseEqKernel, out->size, a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                               size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] == val;
  }
}

void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out) {
  launch_kernel(ScalarEqKernel, out->size, a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t *a, const scalar_t *b,
                              scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] >= b[i];
  }
}

void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  launch_kernel(EwiseGeKernel, out->size, a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                               size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] >= val;
  }
}

void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out) {
  launch_kernel(ScalarGeKernel, out->size, a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t *a, scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = log(a[i]);
  }
}

void EwiseLog(const CudaArray &a, CudaArray *out) {
  launch_kernel(EwiseLogKernel, out->size, a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t *a, scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = exp(a[i]);
  }
}

void EwiseExp(const CudaArray &a, CudaArray *out) {
  launch_kernel(EwiseExpKernel, out->size, a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t *a, scalar_t *out, size_t n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    out[i] = tanh(a[i]);
  }
}

void EwiseTanh(const CudaArray &a, CudaArray *out) {
  launch_kernel(EwiseTanhKernel, out->size, a.ptr, out->ptr, out->size);
}

__global__ void TiledMatMulKernel(scalar_t *a, scalar_t *b, scalar_t *out, int m, int n,
                             int p) {
  /*
   * a: m x n
   * b: n x p
   * out: m x p
   */
  __shared__ scalar_t ms[TILE][TILE];
  __shared__ scalar_t ns[TILE][TILE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE + ty;
  int col = bx * TILE + tx;

  scalar_t pvalues = 0.0;

  for (int ph = 0; ph < (n + TILE - 1) / TILE; ph++) {
    ms[ty][tx] = ((row < m) && (ph * TILE + tx) < n)
                     ? a[row * n + ph * TILE + tx]
                     : 0.0f;
    ns[ty][tx] = ((ph * TILE + ty) < n && (col < p))
                     ? b[(ph * TILE + ty) * p + col]
                     : 0.0f;
    __syncthreads();

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
   * Multiply two (compact) matrices into an output (also compact) matrix.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  int grid_x = (P + TILE - 1) / TILE;
  int grid_y = (M + TILE - 1) / TILE;
  launch_kernel_2d(TiledMatMulKernel, grid_x, grid_y, TILE, TILE,
                   a.ptr, b.ptr, out->ptr, M, N, P);
}

__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t reduce_size, size_t n) {
  /* Each thread would sum across one reduction item */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t offset = i * reduce_size;
    scalar_t reduce_sum = 0;
    for (int j = 0; j < reduce_size; j++) {
      reduce_sum += a[j + offset];
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
   *   reduce_size: size of the dimension to reduce over
   */
  launch_kernel(ReduceSumKernel, out->size, a.ptr, out->ptr, reduce_size, out->size);
}

__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    size_t offset = i * reduce_size;
    scalar_t reduce_max = a[offset];
    for (int j = 1; j < reduce_size; j++) {
      reduce_max = max(reduce_max, a[j + offset]);
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
   *   reduce_size: size of the dimension to reduce over
   */
  launch_kernel(ReduceMaxKernel, out->size, a.ptr, out->ptr, reduce_size, out->size);
}

__device__ int threadIndexToIdx(size_t thread_idx, CudaVec shape, CudaVec strides,
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
   * The CUDA kernel for the compact operation.  This should effectively map a
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
    int idx = threadIndexToIdx(i, shape, strides, offset);
    out[i] = a[idx];
  }
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.
   *
   * Args:
   *   a: non-compact representation of the array
   *   out: compact version of the array to be written
   *   shape: shape of each dimension for a and out
   *   strides: stride of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   *   compact)
   */
  launch_kernel(CompactKernel, out->size, a.ptr, out->ptr, out->size,
                VecToCuda(shape), VecToCuda(strides), offset);
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
  launch_kernel(EwiseSetitemKernel, a.size, a.ptr, out->ptr, a.size,
                VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t *out, size_t n,
                                    CudaVec shape, CudaVec strides,
                                    size_t offset) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[threadIndexToIdx(i, shape, strides, offset)] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray *out,
                   std::vector<int32_t> shape, std::vector<int32_t> strides,
                   size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will not
   *   be the same as out.size, because out is a non-compact subset array);  it
   *   _will_ be the same as the product of items in shape, but convenient to
   *   just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array offset: offset of the out array
   */
  launch_kernel(ScalarSetitemKernel, size, val, out->ptr, size,
                VecToCuda(shape), VecToCuda(strides), offset);
}

} // namespace cuda
} // namespace tiny_pytorch

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace tiny_pytorch;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from GPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t &c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t *host_ptr = (scalar_t *)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0)
      throw std::bad_alloc();
    cudaError_t err =
        cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(err, "cudaMemcpy (device to host)");

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset,
                                 deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
    cudaError_t err = cudaMemcpy(out->ptr, a.request().ptr,
                                 out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err, "cudaMemcpy (host to device)");
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
  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
