// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-ukernels="ukernels=oneDNN" -cse  | FileCheck %s

// Replacement of a triton_cpu.dot operation with triton_cpu.brgemm_execute

// CHECK-LABEL: @test_two_tiles_four_mulf
// CHECK:       %[[LHS_MEMREF:.+]] = triton_cpu.extract_memref %0 : <tensor<16x64xbf16>> -> memref<16x64xbf16, strided<[64, 1]>>
// CHECK-NEXT:  %[[LHS_INDICES:.+]]:2 = triton_cpu.extract_indices %0 : <tensor<16x64xbf16>> -> index, index
// CHECK:       %[[RHS_MEMREF:.+]] = triton_cpu.extract_memref %1 : <tensor<64x32xbf16>> -> memref<64x32xbf16, strided<[32, 1]>>
// CHECK-NEXT:  %[[RHS_INDICES:.+]]:2 = triton_cpu.extract_indices %1 : <tensor<64x32xbf16>> -> index, index
// CHECK:       %[[ACC_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<16x32xf32>
// CHECK:       %[[LHS_SUBVIEW:.+]] = memref.subview %[[LHS_MEMREF]][%[[LHS_INDICES]]#0, %[[LHS_INDICES]]#1] [16, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1], offset: ?>>
// CHECK:       %[[RHS_SUBVIEW:.+]] = memref.subview %[[RHS_MEMREF]][%[[RHS_INDICES]]#0, %[[RHS_INDICES]]#1] [64, 32] [1, 1] : memref<64x32xbf16, strided<[32, 1]>> to memref<64x32xbf16, strided<[32, 1], offset: ?>>
// CHECK:       %[[NONE1:.+]], %[[NONE2:.+]], %[[NONE3:.+]]:2, %[[LHS_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[LHS_SUBVIEW]] : memref<16x64xbf16, strided<[64, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
// CHECK:       %[[NONE4:.+]], %[[NONE5:.+]], %[[NONE6:.+]]:2, %[[RHS_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[RHS_SUBVIEW]] : memref<64x32xbf16, strided<[32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
// CHECK:       %[[NONE7:.+]], %[[NONE8:.+]], %[[NONE0:.+]]:2, %[[ACC_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[ACC_BUF]] : memref<16x32xf32> -> memref<f32>, index, index, index, index, index
// CHECK:       %[[ONEDNN_HANDLE:.+]] = "triton_cpu.brgemm_create"(%c16{{.*}}, %c32{{.*}}, %c64{{.*}}, %c1{{.*}}, %[[LHS_STRIDES]]#0, %[[RHS_STRIDES]]#0, %[[ACC_STRIDES]]#0) <{dtypeA = vector<16x64xbf16>, dtypeB = vector<64x32xbf16>, dtypeC = f32}> : (i64, i64, i64, index, index, index, index) -> index
// CHECK:       %[[BTW:.+]] = arith.constant 2 : i64
// CHECK:       %[[BLOCK:.+]] = arith.muli  %c32{{.*}}, %c64{{.*}} : i64
// CHECK:       %[[BLOCKEDB_SIZE:.+]] = arith.muli %[[BLOCK]], %[[BTW]] : i64
// CHECK:       "triton_cpu.brgemm_execute"(%[[ONEDNN_HANDLE]], %[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]], %[[ACC_BUF]], %c0, %c0, %[[BLOCKEDB_SIZE]], %c1) : (index, memref<16x64xbf16, strided<[64, 1], offset: ?>>, memref<64x32xbf16, strided<[32, 1], offset: ?>>, memref<16x32xf32>, index, index, i64, index) -> ()
// CHECK:       %[[RES:.+]] = vector.transfer_read %[[ACC_BUF]][%c0, %c0], %cst_10 : memref<16x32xf32>, vector<16x32xf32>

#loc = loc(unknown)
module {
  tt.func public @test_two_tiles_four_mulf(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16x32xf32> loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c16_i64 = arith.constant 16 : i64 loc(#loc)
    %c64_i64 = arith.constant 64 : i64 loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %0 = tt.make_tensor_ptr %arg0, [%c16_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xbf16>> loc(#loc)
    %1 = tt.make_tensor_ptr %arg1, [%c64_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>> loc(#loc)
    %2 = tt.make_tensor_ptr %arg2, [%c16_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>> loc(#loc)
    %3 = triton_cpu.extract_memref %0 : <tensor<16x64xbf16>> -> memref<16x64xbf16, strided<[64, 1]>> loc(#loc)
    %4:2 = triton_cpu.extract_indices %0 : <tensor<16x64xbf16>> -> index, index loc(#loc)
    %5 = vector.transfer_read %3[%4#0, %4#1], %cst {in_bounds = [true, true]} : memref<16x64xbf16, strided<[64, 1]>>, vector<16x64xbf16> loc(#loc)
    %6 = triton_cpu.extract_memref %1 : <tensor<64x32xbf16>> -> memref<64x32xbf16, strided<[32, 1]>> loc(#loc)
    %7:2 = triton_cpu.extract_indices %1 : <tensor<64x32xbf16>> -> index, index loc(#loc)
    %8 = vector.transfer_read %6[%7#0, %7#1], %cst {in_bounds = [true, true]} : memref<64x32xbf16, strided<[32, 1]>>, vector<64x32xbf16> loc(#loc)
    %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<16x64xbf16> * vector<64x32xbf16> -> vector<16x32xf32> loc(#loc)
    %10 = triton_cpu.extract_memref %2 : <tensor<16x32xf32>> -> memref<16x32xf32, strided<[32, 1]>> loc(#loc)
    %11:2 = triton_cpu.extract_indices %2 : <tensor<16x32xf32>> -> index, index loc(#loc)
    vector.transfer_write %9, %10[%11#0, %11#1] {in_bounds = [true, true]} : vector<16x32xf32>, memref<16x32xf32, strided<[32, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// More complicated case with a loop that can be replaced with single triton_cpu.brgemm_execute.

// CHECK-LABEL: @test_loop_acc_two_blocks
// CHECK:       %[[LHS_MEMREF:.+]] = triton_cpu.extract_memref %0 : <tensor<64x64xbf16>> -> memref<64x128xbf16, strided<[128, 1]>>
// CHECK-NEXT:  %[[LHS_INDICES:.+]]:2 = triton_cpu.extract_indices %0 : <tensor<64x64xbf16>> -> index, index
// CHECK:       %[[RHS_MEMREF:.+]] = triton_cpu.extract_memref %1 : <tensor<64x32xbf16>> -> memref<128x32xbf16, strided<[32, 1]>>
// CHECK-NEXT:  %[[RHS_INDICES:.+]]:2 = triton_cpu.extract_indices %1 : <tensor<64x32xbf16>> -> index, index
// CHECK:       %[[LOOP_LENGTH:.+]] = arith.subi %c2{{.*}}, %c0{{.*}} : i32
// CHECK:       %[[NUM_BATCHES_INT:.+]] = arith.divui %[[LOOP_LENGTH]], %c1{{.*}} : i32
// CHECK:       %[[NUM_BATCHES:.+]] = arith.index_cast %[[NUM_BATCHES_INT]] : i32 to index
// CHECK:       %[[ACC_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<64x32xf32>
// CHECK:       %[[LHS_SUBVIEW:.+]] = memref.subview %[[LHS_MEMREF]][%[[LHS_INDICES]]#0, %[[LHS_INDICES]]#1] [64, 64] [1, 1] : memref<64x128xbf16, strided<[128, 1]>> to memref<64x64xbf16, strided<[128, 1], offset: ?>>
// CHECK:       %[[RHS_SUBVIEW:.+]] = memref.subview %[[RHS_MEMREF]][%[[RHS_INDICES]]#0, %[[RHS_INDICES]]#1] [64, 32] [1, 1] : memref<128x32xbf16, strided<[32, 1]>> to memref<64x32xbf16, strided<[32, 1], offset: ?>>
// CHECK:       %[[NONE1:.+]], %[[NONE2:.+]], %[[NONE3:.+]]:2, %[[LHS_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[LHS_SUBVIEW]] : memref<64x64xbf16, strided<[128, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
// CHECK:       %[[NONE4:.+]], %[[NONE5:.+]], %[[NONE6:.+]]:2, %[[RHS_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[RHS_SUBVIEW]] : memref<64x32xbf16, strided<[32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
// CHECK:       %[[NONE7:.+]], %[[NONE8:.+]], %[[NONE0:.+]]:2, %[[ACC_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[ACC_BUF]] : memref<64x32xf32> -> memref<f32>, index, index, index, index, index
// CHECK:       %[[ONEDNN_HANDLE:.+]] = "triton_cpu.brgemm_create"(%c64{{.*}}, %c32{{.*}}, %c64{{.*}}, %[[NUM_BATCHES]], %[[LHS_STRIDES]]#0, %[[RHS_STRIDES]]#0, %[[ACC_STRIDES]]#0)  <{dtypeA = vector<64x64xbf16>, dtypeB = vector<64x32xbf16>, dtypeC = f32}> : (i64, i64, i64, index, index, index, index) -> index
// CHECK:       %[[BTW:.+]] = arith.constant 2 : i64
// CHECK:       %[[BLOCK:.+]] = arith.muli  %c32{{.*}}, %c64{{.*}} : i64
// CHECK:       %[[BLOCKEDB_SIZE:.+]] = arith.muli %[[BLOCK]], %[[BTW]] : i64
// CHECK:       %[[STEP:.+]] = arith.index_cast %c0{{.*}} : i32 to index
// CHECK:       %[[OFFSET:.+]] = arith.muli %[[STEP]], %[[LHS_STRIDES]]#0 : index
// CHECK:       %[[SZ:.+]] = arith.addi %c0, %[[OFFSET]] : index
// CHECK:       %[[STEP1:.+]] = arith.index_cast %c64{{.*}} : i32 to index
// CHECK:       %[[OFFSET1:.+]] = arith.muli %[[STEP1]], %[[LHS_STRIDES]]#1 : index
// CHECK:       %[[LHS_STEP_ELEM:.+]] = arith.addi %[[SZ]], %[[OFFSET1]] : index
// CHECK:       %[[CONST:.+]] = arith.constant 2 : index
// CHECK:       %[[LHS_STEP:.+]] = arith.muli %[[LHS_STEP_ELEM]], %[[CONST]] : index
// CHECK:       %[[R_OFFSET:.+]] = arith.muli %[[STEP1]], %[[RHS_STRIDES]]#0 : index
// CHECK:       %[[SZ1:.+]] = arith.addi %c0, %[[R_OFFSET]] : index
// CHECK:       %[[R_OFFSET1:.+]] = arith.muli %[[STEP]], %[[RHS_STRIDES]]#1 : index
// CHECK:       %[[RHS_STEP_ELEM:.+]] = arith.addi %[[SZ1]], %[[R_OFFSET1]] : index
// CHECK:       %[[RHS_STEP:.+]] = arith.muli %[[RHS_STEP_ELEM]], %[[CONST]] : index
// CHECK:       "triton_cpu.brgemm_execute"(%[[ONEDNN_HANDLE]], %[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]], %[[ACC_BUF]], %[[LHS_STEP]], %[[RHS_STEP]], %[[BLOCKEDB_SIZE]], %[[NUM_BATCHES]]) : (index, memref<64x64xbf16, strided<[128, 1], offset: ?>>, memref<64x32xbf16, strided<[32, 1], offset: ?>>, memref<64x32xf32>, index, index, i64, index) -> ()
// CHECK:       %[[RES:.+]] = vector.transfer_read %[[ACC_BUF]][%c0, %c0], %cst_10 {in_bounds = [true, true]} : memref<64x32xf32>, vector<64x32xf32>

#loc = loc(unknown)
module {
  tt.func public @test_loop_acc_two_blocks(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
    %c2_i32 = arith.constant 2 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64x32xf32> loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c64_i64 = arith.constant 64 : i64 loc(#loc)
    %c128_i64 = arith.constant 128 : i64 loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>> loc(#loc)
    %1 = tt.make_tensor_ptr %arg1, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>> loc(#loc)
    %2 = tt.make_tensor_ptr %arg2, [%c64_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf32>> loc(#loc)
    %3:3 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %cst_0, %arg5 = %0, %arg6 = %1) -> (vector<64x32xf32>, !tt.ptr<tensor<64x64xbf16>>, !tt.ptr<tensor<64x32xbf16>>)  : i32 {
      %6 = triton_cpu.extract_memref %arg5 : <tensor<64x64xbf16>> -> memref<64x128xbf16, strided<[128, 1]>> loc(#loc)
      %7:2 = triton_cpu.extract_indices %arg5 : <tensor<64x64xbf16>> -> index, index loc(#loc)
      %8 = vector.transfer_read %6[%7#0, %7#1], %cst {in_bounds = [true, true]} : memref<64x128xbf16, strided<[128, 1]>>, vector<64x64xbf16> loc(#loc)
      %9 = triton_cpu.extract_memref %arg6 : <tensor<64x32xbf16>> -> memref<128x32xbf16, strided<[32, 1]>> loc(#loc)
      %10:2 = triton_cpu.extract_indices %arg6 : <tensor<64x32xbf16>> -> index, index loc(#loc)
      %11 = vector.transfer_read %9[%10#0, %10#1], %cst {in_bounds = [true, true]} : memref<128x32xbf16, strided<[32, 1]>>, vector<64x32xbf16> loc(#loc)
      %12 = triton_cpu.dot %8, %11, %arg4, inputPrecision = ieee : vector<64x64xbf16> * vector<64x32xbf16> -> vector<64x32xf32> loc(#loc)
      %13 = tt.advance %arg5, [%c0_i32, %c64_i32] : <tensor<64x64xbf16>> loc(#loc)
      %14 = tt.advance %arg6, [%c64_i32, %c0_i32] : <tensor<64x32xbf16>> loc(#loc)
      scf.yield %12, %13, %14 : vector<64x32xf32>, !tt.ptr<tensor<64x64xbf16>>, !tt.ptr<tensor<64x32xbf16>> loc(#loc)
    } loc(#loc)
    %4 = triton_cpu.extract_memref %2 : <tensor<64x32xf32>> -> memref<64x32xf32, strided<[32, 1]>> loc(#loc)
    %5:2 = triton_cpu.extract_indices %2 : <tensor<64x32xf32>> -> index, index loc(#loc)
    vector.transfer_write %3#0, %4[%5#0, %5#1] {in_bounds = [true, true]} : vector<64x32xf32>, memref<64x32xf32, strided<[32, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)


// -----

// Case with a loop, that cannot be replaced as a whole and brgemm call should be
// injected in it's body.

// CHECK-LABEL: @test_loop_with_transpose
// CHECK:       %[[LHS_TPTR:.*]] = tt.make_tensor_ptr %arg0, [%17, %18, %c32_i64, %c32_i64], [%19, %c1024_i64, %c32_i64, %c1_i64], [%11, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x32x32xbf16>>
// CHECK:       %[[RHS_TPTR:.*]] = tt.make_tensor_ptr %arg1, [%18, %22, %c16_i64, %c64_i64], [%c1024_i64, %19, %c64_i64, %c1_i64], [%c0_i32, %13, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x16x64xbf16>>
// CHECK:       %[[RES_TPTR:.*]] = tt.make_tensor_ptr %arg2, [%26, %27], [%27, %c1_i64], [%24, %25] {order = array<i32: 1, 0>} : <tensor<32x32xf32>>
// CHECK:       %[[LHS_ALLOCA:.*]] = memref.alloca() {alignment = 64 : i64} : memref<32x32xbf16>
// CHECK:       %[[RHS_ALLOCA:.*]] = memref.alloca() {alignment = 64 : i64} : memref<32x32xbf16>
// CHECK:       %[[RES_ALLOCA:.*]] = memref.alloca() {alignment = 64 : i64} : memref<32x32xf32>
// CHECK:       %31:3 = scf.for %[[IV1:.*]] = %c0_i32 to %30 step %c1_i32 iter_args(%[[RES_IV:.*]] = %cst_0, %[[LHS_IV:.*]] = %[[LHS_TPTR]], %[[RHS_IV:.*]] = %[[RHS_TPTR]]) -> (vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>>)  : i32 {
// CHECK-NEXT:    %[[LHS_MEMREF:.+]] = triton_cpu.extract_memref %[[LHS_IV]] : <tensor<1x1x32x32xbf16>> -> memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>>
// CHECK-NEXT:    %[[LHS_INDICES:.+]]:4 = triton_cpu.extract_indices %[[LHS_IV]] : <tensor<1x1x32x32xbf16>> -> index, index, index, index
// CHECK-NEXT:    %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS_MEMREF]][%[[LHS_INDICES]]#0, %[[LHS_INDICES]]#1, %[[LHS_INDICES]]#2, %[[LHS_INDICES]]#3], %cst {in_bounds = [true, true]} : memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>>, vector<32x32xbf16>
// CHECK-NEXT:    %[[RHS_MEMREF:.+]] = triton_cpu.extract_memref %[[RHS_IV]] : <tensor<1x1x16x64xbf16>> -> memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>>
// CHECK-NEXT:    %[[RHS_INDICES:.+]]:4 = triton_cpu.extract_indices %[[RHS_IV]] : <tensor<1x1x16x64xbf16>> -> index, index, index, index
// CHECK-NEXT:    %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS_MEMREF]][%[[RHS_INDICES]]#0, %[[RHS_INDICES]]#1, %[[RHS_INDICES]]#2, %[[RHS_INDICES]]#3], %cst {in_bounds = [true, true]} : memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>>, vector<16x64xbf16>
// CHECK-NEXT:    %[[LHS_VEC_T:.+]] = vector.transpose %[[LHS_VEC]], [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
// CHECK-NEXT:    %[[RHS_VEC_D_0:.+]], %[[RHS_VEC_D_1:.+]] = vector.deinterleave %[[RHS_VEC]] : vector<16x64xbf16> -> vector<16x32xbf16>
// CHECK-NEXT:    %[[RHS_VEC_D_0_T:.+]] = vector.transpose %[[RHS_VEC_D_0]], [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
// CHECK-NEXT:    %[[RHS_VEC_D_1_T:.+]] = vector.transpose %[[RHS_VEC_D_1]], [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
// CHECK-NEXT:    %[[RHS_VEC_D_T:.+]] = vector.interleave %[[RHS_VEC_D_0_T]], %[[RHS_VEC_D_1_T]] : vector<32x16xbf16> -> vector<32x32xbf16>
// CHECK-NEXT:    %[[RHS_VEC_D:.+]] = vector.transpose %[[RHS_VEC_D_T]], [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
// CHECK:         vector.transfer_write %[[LHS_VEC_T]], %[[LHS_ALLOCA]][%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16>
// CHECK-NEXT:    vector.transfer_write %[[RHS_VEC_D]], %[[RHS_ALLOCA]][%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16>
// CHECK:         %[[LHS_SUBVIEW:.+]] = memref.subview %[[LHS_ALLOCA]][0, 0] [32, 32] [1, 1] : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
// CHECK-NEXT:    %[[RHS_SUBVIEW:.+]] = memref.subview %[[RHS_ALLOCA]][0, 0] [32, 32] [1, 1] : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
// CHECK-NEXT:    %[[NONE1:.+]], %[[NONE2:.+]], %[[NONE3:.+]]:2, %[[LHS_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[LHS_SUBVIEW]] : memref<32x32xbf16, strided<[32, 1]>> -> memref<bf16>, index, index, index, index, index
// CHECK-NEXT:    %[[NONE4:.+]], %[[NONE5:.+]], %[[NONE6:.+]]:2, %[[RHS_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[RHS_SUBVIEW]] : memref<32x32xbf16, strided<[32, 1]>> -> memref<bf16>, index, index, index, index, index
// CHECK-NEXT:    %[[NONE7:.+]], %[[NONE8:.+]], %[[NONE0:.+]]:2, %[[ACC_STRIDES:.+]]:2 = memref.extract_strided_metadata %[[RES_ALLOCA]] : memref<32x32xf32> -> memref<f32>, index, index, index, index, index
// CHECK-NEXT:    %[[ONEDNN_HANDLE:.+]] = "triton_cpu.brgemm_create"(%c32{{.*}}, %c32{{.*}}, %c32{{.*}}, %c1, %[[LHS_STRIDES]]#0, %[[RHS_STRIDES]]#0, %[[ACC_STRIDES]]#0) <{dtypeA = vector<32x32xbf16>, dtypeB = vector<32x32xbf16>, dtypeC = f32}> : (i64, i64, i64, index, index, index, index) -> index
// CHECK-NEXT:    %[[BTW:.+]] = arith.constant 2 : i64
// CHECK-NEXT:    %[[BLOCK:.+]] = arith.muli %c32{{.*}}, %c32{{.*}} : i64
// CHECK-NEXT:    %[[BLOCKEDB_SIZE:.+]] = arith.muli %[[BLOCK]], %[[BTW]] : i64
// CHECK-NEXT:    "triton_cpu.brgemm_execute"(%[[ONEDNN_HANDLE]], %[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]], %[[RES_ALLOCA]], %c0{{.*}}, %c0{{.*}}, %[[BLOCKEDB_SIZE]], %c1) : (index, memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xf32>, index, index, i64, index) -> ()
// CHECK-NEXT:    %[[LHS_IV_UPD:.*]] = tt.advance %[[LHS_IV]], [%c0_i32, %c1_i32, %c0_i32, %c0_i32] : <tensor<1x1x32x32xbf16>>
// CHECK-NEXT:    %[[RHS_IV_UPD:.*]] = tt.advance %[[RHS_IV]], [%c1_i32, %c0_i32, %c0_i32, %c0_i32] : <tensor<1x1x16x64xbf16>>
// CHECK-NEXT:    scf.yield %[[RES_IV]], %[[LHS_IV_UPD]], %[[RHS_IV_UPD]] : vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>>
// CHECK-NEXT:  }
// CHECK:       %[[RES:.+]] = vector.transfer_read %[[RES_ALLOCA]][%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<32x32xf32>, vector<32x32xf32>


#loc = loc(unknown)
module {
  tt.func public @test_loop_with_transpose(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown), %arg3: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg4: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg5: i32 {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
    %c31_i32 = arith.constant 31 : i32 loc(#loc)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32> loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c64_i64 = arith.constant 64 : i64 loc(#loc)
    %c16_i64 = arith.constant 16 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c8_i32 = arith.constant 8 : i32 loc(#loc)
    %0 = tt.get_program_id x : i32 loc(#loc)
    %1 = arith.addi %arg3, %c31_i32 : i32 loc(#loc)
    %2 = arith.divsi %1, %c32_i32 : i32 loc(#loc)
    %3 = arith.addi %arg4, %c31_i32 : i32 loc(#loc)
    %4 = arith.divsi %3, %c32_i32 : i32 loc(#loc)
    %5 = arith.muli %4, %c8_i32 : i32 loc(#loc)
    %6 = arith.divsi %0, %5 : i32 loc(#loc)
    %7 = arith.muli %6, %c8_i32 : i32 loc(#loc)
    %8 = arith.subi %2, %7 : i32 loc(#loc)
    %9 = arith.minsi %8, %c8_i32 : i32 loc(#loc)
    %10 = arith.remsi %0, %9 : i32 loc(#loc)
    %11 = arith.addi %7, %10 : i32 loc(#loc)
    %12 = arith.remsi %0, %5 : i32 loc(#loc)
    %13 = arith.divsi %12, %9 : i32 loc(#loc)
    %14 = arith.divsi %arg3, %c32_i32 : i32 loc(#loc)
    %15 = arith.divsi %arg5, %c32_i32 : i32 loc(#loc)
    %16 = arith.muli %arg5, %c32_i32 : i32 loc(#loc)
    %17 = arith.extsi %14 : i32 to i64 loc(#loc)
    %18 = arith.extsi %15 : i32 to i64 loc(#loc)
    %19 = arith.extsi %16 : i32 to i64 loc(#loc)
    %20 = tt.make_tensor_ptr %arg0, [%17, %18, %c32_i64, %c32_i64], [%19, %c1024_i64, %c32_i64, %c1_i64], [%11, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x32x32xbf16>> loc(#loc)
    %21 = arith.divsi %arg4, %c32_i32 : i32 loc(#loc)
    %22 = arith.extsi %21 : i32 to i64 loc(#loc)
    %23 = tt.make_tensor_ptr %arg1, [%18, %22, %c16_i64, %c64_i64], [%c1024_i64, %19, %c64_i64, %c1_i64], [%c0_i32, %13, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x16x64xbf16>> loc(#loc)
    %24 = arith.muli %11, %c32_i32 : i32 loc(#loc)
    %25 = arith.muli %13, %c32_i32 : i32 loc(#loc)
    %26 = arith.extsi %arg3 : i32 to i64 loc(#loc)
    %27 = arith.extsi %arg4 : i32 to i64 loc(#loc)
    %28 = tt.make_tensor_ptr %arg2, [%26, %27], [%27, %c1_i64], [%24, %25] {order = array<i32: 1, 0>} : <tensor<32x32xf32>> loc(#loc)
    %29 = arith.addi %arg5, %c31_i32 : i32 loc(#loc)
    %30 = arith.divsi %29, %c32_i32 : i32 loc(#loc)
    %31:3 = scf.for %arg6 = %c0_i32 to %30 step %c1_i32 iter_args(%arg7 = %cst_0, %arg8 = %20, %arg9 = %23) -> (vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>>)  : i32 {
      %34 = triton_cpu.extract_memref %arg8 : <tensor<1x1x32x32xbf16>> -> memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>> loc(#loc)
      %35:4 = triton_cpu.extract_indices %arg8 : <tensor<1x1x32x32xbf16>> -> index, index, index, index loc(#loc)
      %36 = vector.transfer_read %34[%35#0, %35#1, %35#2, %35#3], %cst {in_bounds = [true, true]} : memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>>, vector<32x32xbf16> loc(#loc)
      %37 = triton_cpu.extract_memref %arg9 : <tensor<1x1x16x64xbf16>> -> memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>> loc(#loc)
      %38:4 = triton_cpu.extract_indices %arg9 : <tensor<1x1x16x64xbf16>> -> index, index, index, index loc(#loc)
      %39 = vector.transfer_read %37[%38#0, %38#1, %38#2, %38#3], %cst {in_bounds = [true, true]} : memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>>, vector<16x64xbf16> loc(#loc)
      %40 = vector.transpose %36, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16> loc(#loc)
      %res1, %res2 = vector.deinterleave %39 : vector<16x64xbf16> -> vector<16x32xbf16> loc(#loc)
      %41 = vector.transpose %res1, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16> loc(#loc)
      %42 = vector.transpose %res2, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16> loc(#loc)
      %43 = vector.interleave %41, %42 : vector<32x16xbf16> -> vector<32x32xbf16> loc(#loc)
      %44 = vector.transpose %43, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16> loc(#loc)
      %45 = triton_cpu.dot %40, %44, %arg7, inputPrecision = tf32 : vector<32x32xbf16> * vector<32x32xbf16> -> vector<32x32xf32> loc(#loc)
      %46 = tt.advance %arg8, [%c0_i32, %c1_i32, %c0_i32, %c0_i32] : <tensor<1x1x32x32xbf16>> loc(#loc)
      %47 = tt.advance %arg9, [%c1_i32, %c0_i32, %c0_i32, %c0_i32] : <tensor<1x1x16x64xbf16>> loc(#loc)
      scf.yield %45, %46, %47 : vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>> loc(#loc)
    } loc(#loc)
    %32 = triton_cpu.extract_memref %28 : <tensor<32x32xf32>> -> memref<?x?xf32, strided<[?, 1]>> loc(#loc)
    %33:2 = triton_cpu.extract_indices %28 : <tensor<32x32xf32>> -> index, index loc(#loc)
    vector.transfer_write %31#0, %32[%33#0, %33#1] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32, strided<[?, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
