// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avx512bf16 -cse  | FileCheck %s --check-prefixes=AVX512,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avxneconvert -cse  | FileCheck %s --check-prefixes=AVX_NE_CONVERT,ALL

// ALL-LABEL: gemm_looped_avx512
// AVX512:       %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// AVX512:       scf.for %{{.+}} = %c0 to %c32 step %c4
// AVX512:         scf.for %{{.+}} = %c0 to %c64 step %c64
// AVX512:           %{{.+}}:16 = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %c1
// AVX512-SAME:          iter_args(%{{.+}} = %[[ZERO]],
// AVX512-COUNT-16:    x86.avx512.dot
// AVX512:             scf.yield
// AVX512-COUNT-16:  vector.transfer_write

tt.func public @gemm_looped_avx512(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<bf16>, %arg5: !tt.ptr<i32>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<32x64xf32>
  %c7_i32 = arith.constant 7 : i32
  %c8_i32 = arith.constant 8 : i32
  %c64_i64 = arith.constant 64 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c128_i64 = arith.constant 128 : i64
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = arith.divsi %arg6, %c32_i32 : i32
  %1 = arith.divsi %arg7, %c64_i32 : i32
  %2 = arith.divsi %arg8, %c2_i32 : i32
  %3 = arith.addi %2, %c7_i32 : i32
  %4 = arith.divsi %3, %c8_i32 : i32
  %5 = tt.get_program_id x : i32
  %6 = arith.muli %5, %c2_i32 : i32
  %7 = tt.addptr %arg5, %6 : !tt.ptr<i32>, i32
  %8 = tt.load %7 : !tt.ptr<i32>
  %9 = tt.addptr %7, %c1_i32 : !tt.ptr<i32>, i32
  %10 = tt.load %9 : !tt.ptr<i32>
  %11 = arith.muli %arg9, %4 : i32
  %12 = arith.muli %arg8, %c32_i32 : i32
  %13 = arith.extsi %12 : i32 to i64
  %14 = arith.extsi %arg8 : i32 to i64
  %15 = tt.make_tensor_descriptor %arg0, [%0, %2, %c32_i32, %c2_i32], [%13, %c2_i64, %14, %c1_i64] : <bf16>, <1x1x32x2xbf16>
  %16 = arith.muli %arg8, %c64_i32 : i32
  %17 = arith.extsi %16 : i32 to i64
  %18 = tt.make_tensor_descriptor %arg1, [%1, %2, %c1_i32, %c128_i32], [%17, %c128_i64, %c128_i64, %c1_i64] : <bf16>, <1x1x1x128xbf16>
  %19 = arith.muli %arg7, %c32_i32 : i32
  %20 = arith.extsi %19 : i32 to i64
  %21 = tt.make_tensor_descriptor %arg3, [%0, %1, %c32_i32, %c64_i32], [%20, %c2048_i64, %c64_i64, %c1_i64] : <f32>, <1x1x32x64xf32>
  %22 = arith.addi %11, %4 : i32
  %23 = arith.minsi %22, %2 : i32
  %24 = scf.for %arg10 = %11 to %23 step %c1_i32 iter_args(%arg11 = %cst_0) -> (vector<32x64xf32>)  : i32 {
    %29 = triton_cpu.extract_memref %15 : <1x1x32x2xbf16> -> memref<?x?x32x2xbf16, strided<[?, 2, ?, 1]>>
    %30 = arith.index_cast %8 : i32 to index
    %31 = arith.index_cast %arg10 : i32 to index
    %32 = vector.transfer_read %29[%30, %31, %c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?x32x2xbf16, strided<[?, 2, ?, 1]>>, vector<32x2xbf16>
    %33 = triton_cpu.extract_memref %18 : <1x1x1x128xbf16> -> memref<?x?x1x128xbf16, strided<[?, 128, 128, 1]>>
    %34 = arith.index_cast %10 : i32 to index
    %35 = vector.transfer_read %33[%34, %31, %c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?x1x128xbf16, strided<[?, 128, 128, 1]>>, vector<1x128xbf16>
    %res1, %res2 = vector.deinterleave %35 : vector<1x128xbf16> -> vector<1x64xbf16>
    %36 = vector.transpose %res1, [1, 0] : vector<1x64xbf16> to vector<64x1xbf16>
    %37 = vector.transpose %res2, [1, 0] : vector<1x64xbf16> to vector<64x1xbf16>
    %38 = vector.interleave %36, %37 : vector<64x1xbf16> -> vector<64x2xbf16>
    %39 = vector.transpose %38, [1, 0] : vector<64x2xbf16> to vector<2x64xbf16>
    %40 = triton_cpu.dot %32, %39, %arg11, inputPrecision = tf32 : vector<32x2xbf16> * vector<2x64xbf16> -> vector<32x64xf32>
    scf.yield %40 : vector<32x64xf32>
  }
  %25 = vector.shape_cast %24 : vector<32x64xf32> to vector<1x1x32x64xf32>
  %26 = triton_cpu.extract_memref %21 : <1x1x32x64xf32> -> memref<?x?x32x64xf32, strided<[?, 2048, 64, 1]>>
  %27 = arith.index_cast %8 : i32 to index
  %28 = arith.index_cast %10 : i32 to index
  vector.transfer_write %25, %26[%27, %28, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x32x64xf32>, memref<?x?x32x64xf32, strided<[?, 2048, 64, 1]>>
  tt.return
}

// -----

// ALL-LABEL: gemm_looped_avx_ne_convert

// AVX_NE_CONVERT-NOT:   memref.alloca
// AVX_NE_CONVERT:       scf.for %{{.+}} = %c0 to %c32 step %c4
// AVX_NE_CONVERT:         scf.for %{{.+}} = %c0 to %c32 step %c16
// AVX_NE_CONVERT:           %{{.+}}:8 = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %c1
// AVX_NE_CONVERT:             x86.avx.bcst_to_f32.packed
// AVX_NE_CONVERT:             x86.avx.cvt.packed.odd.indexed_to_f32
// AVX_NE_CONVERT:             x86.avx.bcst_to_f32.packed
// AVX_NE_CONVERT:             x86.avx.cvt.packed.even.indexed_to_f32
// AVX_NE_CONVERT:             vector.fma

// In lieu of matching the full sequence:
// AVX_NE_CONVERT-COUNT-7:     vector.fma

// AVX_NE_CONVERT:             scf.yield
// AVX_NE_CONVERT-COUNT-8:  vector.transfer_write

tt.func public @gemm_looped_avx_ne_convert(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<bf16>, %arg5: !tt.ptr<i32>, %arg6: i32, %arg7: i32, %arg8: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c7_i32 = arith.constant 7 : i32
  %c8_i32 = arith.constant 8 : i32
  %c32_i64 = arith.constant 32 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c64_i64 = arith.constant 64 : i64
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = arith.divsi %arg6, %c32_i32 : i32
  %1 = arith.divsi %arg7, %c32_i32 : i32
  %2 = arith.divsi %arg8, %c2_i32 : i32
  %3 = arith.addi %2, %c7_i32 : i32
  %4 = arith.divsi %3, %c8_i32 : i32
  %5 = tt.get_program_id x : i32
  %6 = arith.muli %5, %c2_i32 : i32
  %7 = tt.addptr %arg5, %6 : !tt.ptr<i32>, i32
  %8 = tt.load %7 : !tt.ptr<i32>
  %9 = tt.addptr %7, %c1_i32 : !tt.ptr<i32>, i32
  %10 = tt.load %9 : !tt.ptr<i32>
  %11 = arith.muli %arg8, %c32_i32 : i32
  %12 = arith.extsi %11 : i32 to i64
  %13 = arith.extsi %arg8 : i32 to i64
  %14 = tt.make_tensor_descriptor %arg0, [%0, %2, %c32_i32, %c2_i32], [%12, %c2_i64, %13, %c1_i64] : <bf16>, <1x1x32x2xbf16>
  %15 = tt.make_tensor_descriptor %arg1, [%1, %2, %c1_i32, %c64_i32], [%12, %c64_i64, %c64_i64, %c1_i64] : <bf16>, <1x1x1x64xbf16>
  %16 = arith.muli %arg7, %c32_i32 : i32
  %17 = arith.extsi %16 : i32 to i64
  %18 = tt.make_tensor_descriptor %arg3, [%0, %1, %c32_i32, %c32_i32], [%17, %c1024_i64, %c32_i64, %c1_i64] : <f32>, <1x1x32x32xf32>
  %19 = triton_cpu.extract_memref %18 : <1x1x32x32xf32> -> memref<?x?x32x32xf32, strided<[?, 1024, 32, 1]>>
  %20 = arith.index_cast %8 : i32 to index
  %21 = arith.index_cast %10 : i32 to index
  %22 = vector.transfer_read %19[%20, %21, %c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<?x?x32x32xf32, strided<[?, 1024, 32, 1]>>, vector<32x32xf32>
  %23 = arith.addi %4, %4 : i32
  %24 = arith.minsi %23, %2 : i32
  %25 = scf.for %arg9 = %4 to %24 step %c1_i32 iter_args(%arg10 = %22) -> (vector<32x32xf32>)  : i32 {
    %27 = triton_cpu.extract_memref %14 : <1x1x32x2xbf16> -> memref<?x?x32x2xbf16, strided<[?, 2, ?, 1]>>
    %28 = arith.index_cast %arg9 : i32 to index
    %29 = vector.transfer_read %27[%20, %28, %c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?x32x2xbf16, strided<[?, 2, ?, 1]>>, vector<32x2xbf16>
    %30 = triton_cpu.extract_memref %15 : <1x1x1x64xbf16> -> memref<?x?x1x64xbf16, strided<[?, 64, 64, 1]>>
    %31 = vector.transfer_read %30[%21, %28, %c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?x1x64xbf16, strided<[?, 64, 64, 1]>>, vector<1x64xbf16>
    %res1, %res2 = vector.deinterleave %31 : vector<1x64xbf16> -> vector<1x32xbf16>
    %32 = vector.transpose %res1, [1, 0] : vector<1x32xbf16> to vector<32x1xbf16>
    %33 = vector.transpose %res2, [1, 0] : vector<1x32xbf16> to vector<32x1xbf16>
    %34 = vector.interleave %32, %33 : vector<32x1xbf16> -> vector<32x2xbf16>
    %35 = vector.transpose %34, [1, 0] : vector<32x2xbf16> to vector<2x32xbf16>
    %36 = triton_cpu.dot %29, %35, %arg10, inputPrecision = tf32 : vector<32x2xbf16> * vector<2x32xbf16> -> vector<32x32xf32>
    scf.yield %36 : vector<32x32xf32>
  }
  %26 = vector.shape_cast %25 : vector<32x32xf32> to vector<1x1x32x32xf32>
  vector.transfer_write %26, %19[%20, %21, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x32x32xf32>, memref<?x?x32x32xf32, strided<[?, 1024, 32, 1]>>
  tt.return
}
