#if !defined(XSMM_AVAILABLE)
#error "XSMM ukernel is missing"
#endif // !XSMM_AVAILABLE

#include "libxsmm.h"
#include "libxsmm_utils.h"

#include <stdint.h>
#include <stdio.h>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

extern "C" {

// Helper debug XSMM GEMM shape printer.
static void printXsmmGemmShape(const libxsmm_gemm_shape &gemmShape,
                               FILE *outfile) {
  fprintf(outfile, "M: %d\n", gemmShape.m);
  fprintf(outfile, "N: %d\n", gemmShape.n);
  fprintf(outfile, "K: %d\n", gemmShape.k);
  fprintf(outfile, "lda: %d\n", gemmShape.lda);
  fprintf(outfile, "ldb: %d\n", gemmShape.ldb);
  fprintf(outfile, "ldc: %d\n", gemmShape.ldc);
  fprintf(outfile, "a_in_type: %d\n", gemmShape.a_in_type);
  fprintf(outfile, "b_in_type: %d\n", gemmShape.b_in_type);
  fprintf(outfile, "comp_type: %d\n", gemmShape.comp_type);
  fprintf(outfile, "out_type: %d\n", gemmShape.out_type);
}

// Helper debug XSMM BRGEMM config printer.
static void
printXsmmBrgemmCfg(const libxsmm_gemm_batch_reduce_config &brgemmConfig,
                   FILE *outfile) {
  fprintf(outfile, "br_type: %d\n", brgemmConfig.br_type);
  fprintf(outfile, "br_stride_a_hint: %d\n", brgemmConfig.br_stride_a_hint);
  fprintf(outfile, "br_stride_b_hint: %d\n", brgemmConfig.br_stride_b_hint);
  fprintf(outfile, "br_unroll_hint: %d\n", brgemmConfig.br_unroll_hint);
}

// Returns XSMM compute type for an input data type.
static libxsmm_datatype getXsmmCompType(libxsmm_datatype in_dtype) {
  switch (in_dtype) {
  case LIBXSMM_DATATYPE_F32:
  case LIBXSMM_DATATYPE_F16:
  case LIBXSMM_DATATYPE_BF16:
    return LIBXSMM_DATATYPE_F32;
  default:
    return LIBXSMM_DATATYPE_UNSUPPORTED;
  }
}

EXPORT void *xsmm_brgemm_dispatch(int64_t m, int64_t n, int64_t k, int64_t lda,
                                  int64_t ldb, int64_t ldc,
                                  int64_t stride_a_in_bytes,
                                  int64_t stride_b_in_bytes, int64_t in_dtype,
                                  int64_t out_dtype) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_bitfield l_flags = 0;
  libxsmm_bitfield l_prefetch_flags = 0;

  auto xsmm_in_dtype = static_cast<libxsmm_datatype>(in_dtype);
  auto xsmm_out_dtype = static_cast<libxsmm_datatype>(out_dtype);

  // LIBXSMM col-major - swap A with B.
  libxsmm_gemm_shape l_shape;
  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  // TODO: Add support for mixed precision inputs.
  l_shape.a_in_type = xsmm_in_dtype;
  l_shape.b_in_type = xsmm_in_dtype;
  l_shape.out_type = xsmm_out_dtype;
  // Get accumulator type based on the input data type.
  // Only the final result is converted back to the specified output type.
  l_shape.comp_type = getXsmmCompType(xsmm_in_dtype);
  assert(l_shape.comp_type != LIBXSMM_DATATYPE_UNSUPPORTED &&
         "unsupported input type for comp_type selection");

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;

  // LIBXSMM col-major - swap A with B.
  l_brconfig.br_stride_a_hint = stride_b_in_bytes;
  l_brconfig.br_stride_b_hint = stride_a_in_bytes;
  l_brconfig.br_unroll_hint = 0;

  // Generate the executable JIT code.
  // Ukernels are cached internally by LIBXSMM.
  void *sgemm = (void *)libxsmm_dispatch_brgemm(l_shape, l_flags,
                                                l_prefetch_flags, l_brconfig);
  if (!sgemm) {
    fprintf(stderr, "failed to generate brgemm func\n");
    fprintf(stderr, "in_dtype: %u\n", xsmm_in_dtype);
    fprintf(stderr, "out_dtype: %u\n", xsmm_out_dtype);
    printXsmmGemmShape(l_shape, stderr);
    printXsmmBrgemmCfg(l_brconfig, stderr);
    exit(-1);
  }

  return sgemm;
}

EXPORT void xsmm_brgemm_invoke(void *handle, void *A_ptr, void *B_ptr,
                               void *C_ptr, int64_t num_batches) {
  libxsmm_gemm_param gemm_param;

  unsigned long long num_batches_var = num_batches;
  gemm_param.op.tertiary = (void *)&num_batches_var;

  // LIBXSMM col-major - swap A with B.
  gemm_param.a.primary = B_ptr;
  gemm_param.b.primary = A_ptr;
  gemm_param.c.primary = C_ptr;

  libxsmm_xmmfunction sgemm;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(handle);
  sgemm.gemm(&gemm_param);
}

} // extern C
