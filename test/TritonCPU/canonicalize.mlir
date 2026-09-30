// RUN: triton-opt %s -split-input-file -triton-cpu-canonicalize | FileCheck %s

// Fold transfer read and shape cast.

// CHECK-LABEL: @fold_transfer_read_shape_cast
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK:       vector.transfer_write %[[VAL]]

module {
  tt.func public @fold_transfer_read_shape_cast(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %in_p = tt.make_tensor_descriptor %arg0, [%c2_i32, %c2_i32, %c16_i32, %c16_i32], [%c512_i64, %c256_i64, %c16_i64, %c1_i64] : <bf16>, <1x1x16x16xbf16>
    %out_p = tt.make_tensor_descriptor %arg1, [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <bf16>, <16x16xbf16>
    %memref1 = triton_cpu.extract_memref %in_p : <1x1x16x16xbf16> -> memref<2x2x16x16xbf16, strided<[512, 256, 16, 1]>>
    %val1 = vector.transfer_read %memref1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x2x16x16xbf16, strided<[512, 256, 16, 1]>>, vector<1x1x16x16xbf16>
    %val2 = vector.shape_cast %val1 : vector<1x1x16x16xbf16> to vector<16x16xbf16>
    %memref2 = triton_cpu.extract_memref %out_p : <16x16xbf16> -> memref<16x16xbf16, strided<[16, 1]>>
    vector.transfer_write %val2, %memref2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xbf16>, memref<16x16xbf16, strided<[16, 1]>>
    tt.return
  }
}

// -----

// CHECK-LABEL: @fold_shape_cast_into_transfer_read
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK-SAME:    vector<8x16x32xf32>
// CHECK:       vector.transfer_write %[[VAL]]

tt.func public @fold_shape_cast_into_transfer_read(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.get_program_id x : i32
  %1 = tt.get_program_id y : i32
  %2 = tt.get_program_id z : i32
  %3 = arith.muli %arg3, %arg4 : i32
  %4 = arith.muli %3, %arg5 : i32
  %5 = arith.muli %4, %arg6 : i32
  %6 = arith.muli %arg4, %arg5 : i32
  %7 = arith.muli %6, %arg6 : i32
  %8 = arith.muli %arg5, %arg6 : i32
  %9 = arith.extsi %5 : i32 to i64
  %10 = arith.extsi %7 : i32 to i64
  %11 = arith.extsi %8 : i32 to i64
  %12 = arith.extsi %arg6 : i32 to i64
  %13 = tt.make_tensor_descriptor %arg0, [%arg2, %arg3, %arg4, %arg5, %arg6], [%9, %10, %11, %12, %c1_i64] : <f32>, <1x1x8x16x32xf32>
  %14 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5, %arg6], [%11, %12, %c1_i64] : <f32>, <8x16x32xf32>
  %15 = triton_cpu.extract_memref %13 : <1x1x8x16x32xf32> -> memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1]>>
  %16 = arith.index_cast %0 : i32 to index
  %17 = arith.index_cast %1 : i32 to index
  %18 = arith.index_cast %2 : i32 to index
  %19 = vector.transfer_read %15[%c2, %c3, %16, %17, %18], %cst {in_bounds = [true, true, true, true, true]} : memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1]>>, vector<1x1x8x16x32xf32>
  %20 = vector.shape_cast %19 : vector<1x1x8x16x32xf32> to vector<8x16x32xf32>
  %21 = triton_cpu.extract_memref %14 : <8x16x32xf32> -> memref<?x?x?xf32, strided<[?, ?, 1]>>
  vector.transfer_write %20, %21[%16, %17, %18] {in_bounds = [true, true, true]} : vector<8x16x32xf32>, memref<?x?x?xf32, strided<[?, ?, 1]>>
  tt.return
}

// -----

// CHECK-LABEL: @fold_shape_cast_into_transfer_write
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK-SAME:    vector<64xf32>
// CHECK:       vector.transfer_write %[[VAL]]

tt.func public @fold_shape_cast_into_transfer_write(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.get_program_id x : i32
  %1 = tt.get_program_id y : i32
  %2 = tt.get_program_id z : i32
  %3 = tt.make_tensor_descriptor %arg0, [%arg6], [%c1_i64] : <f32>, <64xf32>
  %4 = arith.muli %arg3, %arg4 : i32
  %5 = arith.muli %4, %arg5 : i32
  %6 = arith.muli %5, %arg6 : i32
  %7 = arith.muli %arg4, %arg5 : i32
  %8 = arith.muli %7, %arg6 : i32
  %9 = arith.muli %arg5, %arg6 : i32
  %10 = arith.extsi %6 : i32 to i64
  %11 = arith.extsi %8 : i32 to i64
  %12 = arith.extsi %9 : i32 to i64
  %13 = arith.extsi %arg6 : i32 to i64
  %14 = tt.make_tensor_descriptor %arg1, [%arg2, %arg3, %arg4, %arg5, %arg6], [%10, %11, %12, %13, %c1_i64] : <f32>, <1x1x1x1x64xf32>
  %15 = triton_cpu.extract_memref %3 : <64xf32> -> memref<?xf32, strided<[1]>>
  %16 = arith.index_cast %2 : i32 to index
  %17 = vector.transfer_read %15[%16], %cst {in_bounds = [true]} : memref<?xf32, strided<[1]>>, vector<64xf32>
  %18 = vector.shape_cast %17 : vector<64xf32> to vector<1x1x1x1x64xf32>
  %19 = triton_cpu.extract_memref %14 : <1x1x1x1x64xf32> -> memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1]>>
  %20 = arith.index_cast %0 : i32 to index
  %21 = arith.index_cast %1 : i32 to index
  vector.transfer_write %18, %19[%c2, %c3, %20, %21, %16] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xf32>, memref<?x?x?x?x?xf32, strided<[?, ?, ?, ?, 1]>>
  tt.return
}

// -----

// CHECK:       #[[PERM_MAP:.+]] = affine_map<(d0, d1) -> (0, d0, d1)>
// CHECK:       @bcast_read_last_dim
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK-SAME:    permutation_map = #[[PERM_MAP]]
// CHECK-SAME:    vector<32x16x8xf32>
// CHECK-NOT:   vector.broadcast
// CHECK:       vector.transfer_write %[[VAL]]

tt.func public @bcast_read_last_dim(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.get_program_id x : i32
  %1 = tt.get_program_id y : i32
  %2 = tt.get_program_id z : i32
  %3 = arith.extsi %arg4 : i32 to i64
  %4 = tt.make_tensor_descriptor %arg0, [%arg3, %arg4], [%3, %c1_i64] : <f32>, <16x8xf32>
  %5 = arith.muli %arg3, %arg4 : i32
  %6 = arith.extsi %5 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg2, %arg3, %arg4], [%6, %3, %c1_i64] : <f32>, <32x16x8xf32>
  %8 = triton_cpu.extract_memref %4 : <16x8xf32> -> memref<?x?xf32, strided<[?, 1]>>
  %9 = arith.index_cast %1 : i32 to index
  %10 = arith.index_cast %2 : i32 to index
  %11 = vector.transfer_read %8[%9, %10], %cst {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1]>>, vector<16x8xf32>
  %12 = vector.broadcast %11 : vector<16x8xf32> to vector<32x16x8xf32>
  %13 = triton_cpu.extract_memref %7 : <32x16x8xf32> -> memref<?x?x?xf32, strided<[?, ?, 1]>>
  %14 = arith.index_cast %0 : i32 to index
  vector.transfer_write %12, %13[%14, %9, %10] {in_bounds = [true, true, true]} : vector<32x16x8xf32>, memref<?x?x?xf32, strided<[?, ?, 1]>>
  tt.return
}

// -----

// CHECK:       #[[PERM_MAP:.+]] = affine_map<(d0) -> (d0, 0, 0)>
// CHECK:       @bcast_read_first_dim
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK-SAME:    permutation_map = #[[PERM_MAP]]
// CHECK-SAME:    vector<32x16x8xf32>
// CHECK-NOT:   vector.shape_cast
// CHECK-NOT:   vector.broadcast
// CHECK:       vector.transfer_write %[[VAL]]

tt.func public @bcast_read_first_dim(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.get_program_id x : i32
  %1 = tt.get_program_id y : i32
  %2 = tt.get_program_id z : i32
  %3 = tt.make_tensor_descriptor %arg0, [%arg2], [%c1_i64] : <f32>, <32xf32>
  %4 = arith.muli %arg3, %arg4 : i32
  %5 = arith.extsi %4 : i32 to i64
  %6 = arith.extsi %arg4 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg2, %arg3, %arg4], [%5, %6, %c1_i64] : <f32>, <32x16x8xf32>
  %8 = triton_cpu.extract_memref %3 : <32xf32> -> memref<?xf32, strided<[1]>>
  %9 = arith.index_cast %0 : i32 to index
  %10 = vector.transfer_read %8[%9], %cst {in_bounds = [true]} : memref<?xf32, strided<[1]>>, vector<32xf32>
  %11 = vector.shape_cast %10 {axis = 2 : i32} : vector<32xf32> to vector<32x1x1xf32>
  %12 = vector.broadcast %11 : vector<32x1x1xf32> to vector<32x16x8xf32>
  %13 = triton_cpu.extract_memref %7 : <32x16x8xf32> -> memref<?x?x?xf32, strided<[?, ?, 1]>>
  %14 = arith.index_cast %1 : i32 to index
  %15 = arith.index_cast %2 : i32 to index
  vector.transfer_write %12, %13[%9, %14, %15] {in_bounds = [true, true, true]} : vector<32x16x8xf32>, memref<?x?x?xf32, strided<[?, ?, 1]>>
  tt.return
}

// -----

// CHECK:       #[[PERM_MAP:.+]] = affine_map<(d0) -> (0, d0)>
// CHECK:       @bcast_read_ext
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK-SAME:    permutation_map = #[[PERM_MAP]]
// CHECK-SAME:    vector<16x8xbf16>
// CHECK:       %[[VAL_EXT:.+]] = arith.extf %[[VAL]] : vector<16x8xbf16> to vector<16x8xf32>
// CHECK:       vector.transfer_write %[[VAL_EXT]]

tt.func public @bcast_read_ext(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.get_program_id y : i32
  %1 = tt.get_program_id z : i32
  %2 = tt.make_tensor_descriptor %arg0, [%arg4], [%c1_i64] : <bf16>, <8xbf16>
  %3 = arith.extsi %arg4 : i32 to i64
  %4 = tt.make_tensor_descriptor %arg1, [%arg3, %arg4], [%3, %c1_i64] : <f32>, <16x8xf32>
  %5 = triton_cpu.extract_memref %2 : <8xbf16> -> memref<?xbf16, strided<[1]>>
  %6 = arith.index_cast %1 : i32 to index
  %7 = vector.transfer_read %5[%6], %cst {in_bounds = [true]} : memref<?xbf16, strided<[1]>>, vector<8xbf16>
  %8 = arith.extf %7 : vector<8xbf16> to vector<8xf32>
  %9 = vector.broadcast %8 : vector<8xf32> to vector<16x8xf32>
  %10 = triton_cpu.extract_memref %4 : <16x8xf32> -> memref<?x?xf32, strided<[?, 1]>>
  %11 = arith.index_cast %0 : i32 to index
  vector.transfer_write %9, %10[%11, %6] {in_bounds = [true, true]} : vector<16x8xf32>, memref<?x?xf32, strided<[?, 1]>>
  tt.return
}

// -----

// CHECK:       #[[PERM_MAP:.+]] = affine_map<(d0) -> (0, d0)>
// CHECK:       @bcast_read_ext_int
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK-SAME:    permutation_map = #[[PERM_MAP]]
// CHECK-SAME:    vector<16x8xi8>
// CHECK:       %[[VAL_EXT:.+]] = arith.extsi %[[VAL]] : vector<16x8xi8> to vector<16x8xi32>
// CHECK:       vector.transfer_write %[[VAL_EXT]]

tt.func public @bcast_read_ext_int(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i32>, %arg2: i32, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0 : i8
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.get_program_id y : i32
  %1 = tt.get_program_id z : i32
  %2 = tt.make_tensor_descriptor %arg0, [%arg4], [%c1_i64] : <i8>, <8xi8>
  %3 = arith.extsi %arg4 : i32 to i64
  %4 = tt.make_tensor_descriptor %arg1, [%arg3, %arg4], [%3, %c1_i64] : <i32>, <16x8xi32>
  %5 = triton_cpu.extract_memref %2 : <8xi8> -> memref<?xi8, strided<[1]>>
  %6 = arith.index_cast %1 : i32 to index
  %7 = vector.transfer_read %5[%6], %cst {in_bounds = [true]} : memref<?xi8, strided<[1]>>, vector<8xi8>
  %8 = arith.extsi %7 : vector<8xi8> to vector<8xi32>
  %9 = vector.broadcast %8 : vector<8xi32> to vector<16x8xi32>
  %10 = triton_cpu.extract_memref %4 : <16x8xi32> -> memref<?x?xi32, strided<[?, 1]>>
  %11 = arith.index_cast %0 : i32 to index
  vector.transfer_write %9, %10[%11, %6] {in_bounds = [true, true]} : vector<16x8xi32>, memref<?x?xi32, strided<[?, 1]>>
  tt.return
}

// -----

// Test that the original order (extend first, then broadcast) is preserved if the broadcast cannot be folded into the read.

// CHECK-NOT:   affine_map
// CHECK:       @negative_bcast_read_ext
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK-SAME:    vector<8xbf16>
// CHECK:       %[[VAL_EXT:.+]] = arith.extf %[[VAL]] : vector<8xbf16> to vector<8xf32>
// CHECK:       %[[VAL_BCAST:.+]] = vector.broadcast %[[VAL_EXT]] : vector<8xf32> to vector<16x8xf32>
// CHECK:       vector.transfer_write %[[VAL_BCAST]]

tt.func public @negative_bcast_read_ext(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.get_program_id y : i32
  %1 = tt.get_program_id z : i32
  %2 = tt.make_tensor_descriptor %arg0, [%arg4], [%c1_i64] : <bf16>, <8xbf16>
  %3 = arith.extsi %arg4 : i32 to i64
  %4 = tt.make_tensor_descriptor %arg1, [%arg3, %arg4], [%3, %c1_i64] : <f32>, <16x8xf32>
  %5 = triton_cpu.extract_memref %2 : <8xbf16> -> memref<?xbf16, strided<[1]>>
  %6 = arith.index_cast %1 : i32 to index
  %7 = vector.transfer_read %5[%6], %cst {in_bounds = [false]} : memref<?xbf16, strided<[1]>>, vector<8xbf16>
  %8 = arith.extf %7 : vector<8xbf16> to vector<8xf32>
  %9 = vector.broadcast %8 : vector<8xf32> to vector<16x8xf32>
  %10 = triton_cpu.extract_memref %4 : <16x8xf32> -> memref<?x?xf32, strided<[?, 1]>>
  %11 = arith.index_cast %0 : i32 to index
  vector.transfer_write %9, %10[%11, %6] {in_bounds = [true, true]} : vector<16x8xf32>, memref<?x?xf32, strided<[?, 1]>>
  tt.return
}
