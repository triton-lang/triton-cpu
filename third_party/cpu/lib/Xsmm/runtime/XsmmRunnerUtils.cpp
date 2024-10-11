//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "XsmmRunnerUtils.h"
#include "libxsmm.h" // NOLINT [build/include_subdir]
#include "libxsmm_utils.h"

// Helper function prototypes.
static void printXsmmStruct(const libxsmm_gemm_shape &gemmShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_meltw_unary_shape &unaryShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_meltw_binary_shape &binaryShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_gemm_batch_reduce_config &brgemmShape,
                            FILE *outfile = stderr);

static bool hasImplicitComputeDtypeUnary(const libxsmm_meltw_unary_type dtype) {
  switch (dtype) {
  // Zero
  case LIBXSMM_MELTW_TYPE_UNARY_XOR:
  // Copy
  case LIBXSMM_MELTW_TYPE_UNARY_IDENTITY:
  // Transpose
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT:
  // VNNI2
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2:
  // VNNI4
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM:
  case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2:
    return true;
  default:
    return false;
  }
}

namespace {

void *get_base_ptr(const libxsmm_datatype dType, void *alignedPtr,
                   int64_t offset) {
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *base_ptr = (float *)alignedPtr + offset;
    return (void *)base_ptr;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    bf16 *base_ptr = (bf16 *)alignedPtr + offset;
    return (void *)base_ptr;
  }
  fprintf(stderr, "Unhandled data type in get_data_pointer_from_memref_desc:%d",
          dType);
  return nullptr;
}

} // namespace

extern "C" void xsmm_gemm_invoke(const libxsmm_datatype dType, int64_t addr,
                                 void *alignedPtrA, int64_t offsetA,
                                 void *alignedPtrB, int64_t offsetB,
                                 void *alignedPtrC, int64_t offsetC) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;

  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = get_base_ptr(dType, alignedPtrB, offsetB);
  gemm_param.b.primary = get_base_ptr(dType, alignedPtrA, offsetA);
  gemm_param.c.primary = get_base_ptr(dType, alignedPtrC, offsetC);

  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t xsmm_gemm_dispatch(const libxsmm_datatype dtype, int64_t m,
                                      int64_t n, int64_t k, int64_t lda,
                                      int64_t ldb, int64_t ldc,
                                      const libxsmm_gemm_flags flags) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "ldb: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";

  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = flags;
  libxsmm_bitfield l_prefetch_flags = 0;

  // See:
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  // LIBXSMM col-major change m with n.
  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb;
  l_shape.ldb = lda;
  l_shape.ldc = ldc;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  l_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;

  auto sgemm = libxsmm_dispatch_gemm(l_shape, l_flags, l_prefetch_flags);
  if (!sgemm) {
    fprintf(stderr, "failed to generate matmul func\n");
    fprintf(stderr, "dtype: %u\n", dtype);
    printXsmmStruct(l_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t
xsmm_unary_dispatch(const libxsmm_meltw_unary_type op_type,
                    const libxsmm_datatype dtype, int64_t m, int64_t n,
                    int64_t ldi, int64_t ldo,
                    const libxsmm_meltw_unary_flags unary_flags) {
  // std::cout << "ldi: " << ldi << "\n";
  // std::cout << "ldo: " << ldo << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "type: " << type << "\n";
  // std::cout << "bcast_type: " << bcast_type << "\n";

  libxsmm_meltw_unary_shape unary_shape;
  // Row major to col major swap m with n.
  unary_shape.m = static_cast<libxsmm_blasint>(n);
  unary_shape.n = static_cast<libxsmm_blasint>(m);
  unary_shape.in0_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  // Copy and Zero should remain in BF16 to avoid useless up/down casts
  auto force_fp32 = (dtype == LIBXSMM_DATATYPE_BF16 &&
                     !hasImplicitComputeDtypeUnary(op_type));
  unary_shape.comp_type = force_fp32 ? LIBXSMM_DATATYPE_F32 : dtype;
  unary_shape.out_type = dtype;
  unary_shape.ldi = static_cast<libxsmm_blasint>(ldi);
  unary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_unary kernel =
      libxsmm_dispatch_meltw_unary(op_type, unary_shape, unary_flags);
  if (!kernel) {
    fprintf(stderr, "failed to generate unary func\n");
    fprintf(stderr, "op_type: %u\n", op_type);
    fprintf(stderr, "flags: %u\n", unary_flags);
    printXsmmStruct(unary_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t
xsmm_binary_dispatch(const libxsmm_meltw_binary_type op_type,
                     const libxsmm_datatype dtype, int64_t m, int64_t n,
                     int64_t ldiLhs, int64_t ldiRhs, int64_t ldo,
                     const libxsmm_meltw_binary_flags flags) {
  libxsmm_meltw_binary_shape binary_shape;
  // Row major to col major swap m with n.
  binary_shape.m = static_cast<libxsmm_blasint>(n);
  binary_shape.n = static_cast<libxsmm_blasint>(m);
  binary_shape.in0_type = dtype;
  binary_shape.in1_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  binary_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;
  binary_shape.out_type = dtype;
  binary_shape.ldi = static_cast<libxsmm_blasint>(ldiLhs);
  binary_shape.ldi2 = static_cast<libxsmm_blasint>(ldiRhs);
  binary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_binary kernel =
      libxsmm_dispatch_meltw_binary(op_type, binary_shape, flags);
  if (!kernel) {
    fprintf(stderr, "failed to generate binary func\n");
    fprintf(stderr, "op_type: %u\n", op_type);
    fprintf(stderr, "flags: %u\n", flags);
    printXsmmStruct(binary_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t xsmm_intel_amx_tile_config_dispatch(
    const libxsmm_datatype dtype, int64_t m, int64_t n, int64_t k, int64_t lda,
    int64_t ldb, int64_t ldc, int64_t stride_a, int64_t stride_b,
    const libxsmm_gemm_flags flags) {
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_cfg_flags = flags;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb;
  l_shape.ldb = lda;
  l_shape.ldc = ldc;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  l_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;

  auto sgemm = libxsmm_dispatch_tilecfg_gemm(l_shape, l_cfg_flags);
  if (!sgemm) {
    fprintf(stderr, "failed to generate tileconfig func\n");
    fprintf(stderr, "dtype: %u\n", dtype);
    fprintf(stderr, "flags: %u\n", flags);
    printXsmmStruct(l_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void xsmm_unary_invoke(const libxsmm_datatype dType, int64_t addr,
                                  void *alignedPtrIn, int64_t offsetIn,
                                  void *alignedPtrOut, int64_t offsetOut) {
  libxsmm_meltw_unary_param param;

  param.in.primary = get_base_ptr(dType, alignedPtrIn, offsetIn);
  param.out.primary = get_base_ptr(dType, alignedPtrOut, offsetOut);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  kernel(&param);
}

extern "C" void xsmm_binary_invoke(const libxsmm_datatype dType, int64_t addr,
                                   void *alignedPtrLhs, int64_t offsetLhs,
                                   void *alignedPtrRhs, int64_t offsetRhs,
                                   void *alignedPtrOut, int64_t offsetOut) {
  libxsmm_meltw_binary_param param;

  param.in0.primary = get_base_ptr(dType, alignedPtrLhs, offsetLhs);
  param.in1.primary = get_base_ptr(dType, alignedPtrRhs, offsetRhs);
  param.out.primary = get_base_ptr(dType, alignedPtrOut, offsetOut);

  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(addr);
  kernel(&param);
}

extern "C" void xsmm_unary_scalar_invoke(const libxsmm_datatype dType,
                                         int64_t addr, float input,
                                         void *alignedOut, int64_t offsetOut) {
  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;

  param.in.primary = (void *)&input;
  param.out.primary = get_base_ptr(dType, alignedOut, offsetOut);
  kernel(&param);
}

extern "C" void xsmm_brgemm_invoke(const libxsmm_datatype dType, int64_t addr,
                                   void *alignedPtrA, int64_t offsetA,
                                   void *alignedPtrB, int64_t offsetB,
                                   void *alignedPtrC, int64_t offsetC,
                                   int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;

  unsigned long long numBatchesVar = numBatches;
  gemm_param.op.tertiary = (void *)&numBatchesVar;

  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = get_base_ptr(dType, alignedPtrB, offsetB);
  gemm_param.b.primary = get_base_ptr(dType, alignedPtrA, offsetA);
  gemm_param.c.primary = get_base_ptr(dType, alignedPtrC, offsetC);

  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t xsmm_brgemm_dispatch(const libxsmm_datatype dtype, int64_t m,
                                        int64_t n, int64_t k, int64_t lda,
                                        int64_t ldb, int64_t ldc,
                                        int64_t stride_a, int64_t stride_b,
                                        const libxsmm_gemm_flags flags) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "lbd: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = flags;
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  l_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  auto typeSize = dtype == LIBXSMM_DATATYPE_F32 ? sizeof(float) : sizeof(bf16);
  l_brconfig.br_stride_a_hint = stride_b * typeSize;
  l_brconfig.br_stride_b_hint = stride_a * typeSize;
  l_brconfig.br_unroll_hint = 0;

  auto sgemm =
      libxsmm_dispatch_brgemm(l_shape, l_flags, l_prefetch_flags, l_brconfig);
  if (!sgemm) {
    fprintf(stderr, "failed to generate brgemm func\n");
    fprintf(stderr, "dtype: %u\n", dtype);
    printXsmmStruct(l_shape);
    printXsmmStruct(l_brconfig);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void xsmm_fused_brgemm_invoke(const libxsmm_datatype dType,
                                         int64_t addr, void *alignedPtrA,
                                         int64_t offsetA, void *alignedPtrB,
                                         int64_t offsetB, void *alignedPtrC,
                                         int64_t offsetC, void *alignedPtrD,
                                         int64_t offsetD, int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;

  unsigned long long numBatchesVar = numBatches;
  gemm_param.op.tertiary = (void *)&numBatchesVar;

  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = get_base_ptr(dType, alignedPtrB, offsetB);
  gemm_param.b.primary = get_base_ptr(dType, alignedPtrA, offsetA);
  gemm_param.c.primary = get_base_ptr(dType, alignedPtrC, offsetC);
  gemm_param.d.primary = get_base_ptr(dType, alignedPtrD, offsetD);

  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(addr);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t
xsmm_fused_brgemm_dispatch(const libxsmm_datatype data_type, int64_t m,
                           int64_t n, int64_t k, int64_t lda, int64_t ldb,
                           int64_t ldc, int64_t stride_a, int64_t stride_b,
                           const libxsmm_gemm_flags gemm_flags,
                           const libxsmm_meltw_unary_flags unary_flags,
                           const libxsmm_meltw_unary_type unary_op_type,
                           const libxsmm_meltw_binary_flags binary_flags,
                           const libxsmm_meltw_binary_type binary_op_type) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "lbd: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = gemm_flags;
  libxsmm_bitfield l_prefetch_flags = 0;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = data_type;
  l_shape.b_in_type = data_type;
  l_shape.out_type = data_type;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  l_shape.comp_type =
      data_type == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : data_type;

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  auto typeSize =
      data_type == LIBXSMM_DATATYPE_F32 ? sizeof(float) : sizeof(bf16);
  l_brconfig.br_stride_a_hint = stride_b * typeSize;
  l_brconfig.br_stride_b_hint = stride_a * typeSize;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = unary_op_type;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = data_type;

  l_postops.d_binary_flags = binary_flags;
  l_postops.d_binary_type = binary_op_type;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext(l_shape, l_flags, l_prefetch_flags,
                                           l_brconfig, l_argops, l_postops);
  if (!sgemm) {
    fprintf(stderr, "failed to generate fused brgemm func\n");
    fprintf(stderr, "data_type: %u\n", data_type);
    printXsmmStruct(l_shape);
    printXsmmStruct(l_brconfig);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_intel_amx_tile_config_invoke(const libxsmm_datatype dType, int64_t addr,
                                  void *tileState, int64_t offset) {
  libxsmm_xmmfunction cfg_tr;

  libxsmm_tilecfg_state *l_tilestate =
      reinterpret_cast<libxsmm_tilecfg_state *>(tileState);

  cfg_tr.tilecfg = reinterpret_cast<libxsmm_tilecfgfunction>(addr);
  cfg_tr.tilecfg(l_tilestate);
}

static void printXsmmStruct(const libxsmm_gemm_shape &gemmShape,
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

static void printXsmmStruct(const libxsmm_meltw_unary_shape &unaryShape,
                            FILE *outfile) {
  fprintf(outfile, "M: %d\n", unaryShape.m);
  fprintf(outfile, "N: %d\n", unaryShape.n);
  fprintf(outfile, "in0_type: %d\n", unaryShape.in0_type);
  fprintf(outfile, "comp_type: %d\n", unaryShape.comp_type);
  fprintf(outfile, "out_type: %d\n", unaryShape.out_type);
  fprintf(outfile, "ldi: %d\n", unaryShape.ldi);
  fprintf(outfile, "ldo: %d\n", unaryShape.ldo);
}

static void printXsmmStruct(const libxsmm_meltw_binary_shape &binaryShape,
                            FILE *outfile) {
  fprintf(outfile, "M: %d\n", binaryShape.m);
  fprintf(outfile, "N: %d\n", binaryShape.n);
  fprintf(outfile, "in0_type: %d\n", binaryShape.in0_type);
  fprintf(outfile, "in1_type: %d\n", binaryShape.in1_type);
  fprintf(outfile, "comp_type: %d\n", binaryShape.comp_type);
  fprintf(outfile, "out_type: %d\n", binaryShape.out_type);
  fprintf(outfile, "ldi: %d\n", binaryShape.ldi);
  fprintf(outfile, "ldi2: %d\n", binaryShape.ldi2);
  fprintf(outfile, "ldo: %d\n", binaryShape.ldo);
}

static void
printXsmmStruct(const libxsmm_gemm_batch_reduce_config &brgemmConfig,
                FILE *outfile) {
  fprintf(outfile, "br_type: %d\n", brgemmConfig.br_type);
  fprintf(outfile, "br_stride_a_hint: %d\n", brgemmConfig.br_stride_a_hint);
  fprintf(outfile, "br_stride_b_hint: %d\n", brgemmConfig.br_stride_b_hint);
  fprintf(outfile, "br_unroll_hint: %d\n", brgemmConfig.br_unroll_hint);
}
