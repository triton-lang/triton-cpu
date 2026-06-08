// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=amx-bf16,amx-int8 -cse  | FileCheck %s --check-prefixes=AMX,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avx512bf16 -cse  | FileCheck %s --check-prefixes=AVX512,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avx10.2 -cse  | FileCheck %s --check-prefixes=AVX10_2,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avxneconvert -cse  | FileCheck %s --check-prefixes=AVX_NE_CONVERT,ALL

// AMX bf16 flat/online-packing vector.contract inside of an accumulator loop

// ALL-LABEL: @gemm_amx_bf16

// Tile registers for accumulation
// AMX-COUNT-4: x86.amx.tile_zero : !x86.amx.tile<16x16xf32>

// Prologue
// AMX:         scf.for %arg{{.+}} = %c0 to %c32 step %c2
// AMX:           vector.shuffle %{{.+}}, %{{.+}} [0, 32, 1, 33,
// AMX:           vector.shuffle %{{.+}}, %{{.+}} [4, 36, 5, 37,

// Main pipeline
// AMX:         %{{.+}}:4 = scf.for %arg{{.+}} = %c0 to %{{.+}} step %c32 iter_args(%arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}) ->
// AMX-SAME:      (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>)
// AMX-COUNT-4:   x86.amx.tile_mulf

// Epilogue
// AMX:         %{{.+}}:4 = scf.for %arg{{.+}} = %{{.+}} to %{{.+}} step %c32 iter_args(%arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}) ->
// AMX-SAME:      (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>)

// Store results back to memory
// AMX-COUNT-4: x86.amx.tile_store

// Shuffle results
// AMX:         scf.for %arg{{.+}} = %c0 to %c32 step %c1

tt.func public @gemm_amx_bf16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = arith.extsi %arg5 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] : <bf16>, <32x32xbf16>
  %2 = arith.extsi %arg4 : i32 to i64
  %3 = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%2, %c1_i64] : <bf16>, <32x32xbf16>
  %4 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%2, %c1_i64] : <f32>, <32x32xf32>
  scf.for %arg6 = %c0_i32 to %arg3 step %c32_i32  : i32 {
    scf.for %arg7 = %c0_i32 to %arg4 step %c32_i32  : i32 {
      %5 = triton_cpu.extract_memref %4 : <32x32xf32> -> memref<?x?xf32, strided<[?, 1]>>
      %6 = arith.index_cast %arg6 : i32 to index
      %7 = arith.index_cast %arg7 : i32 to index
      %8 = vector.transfer_read %5[%6, %7], %cst_0 {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1]>>, vector<32x32xf32>
      %9 = scf.for %arg8 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg9 = %8) -> (vector<32x32xf32>)  : i32 {
        %10 = triton_cpu.extract_memref %1 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
        %11 = arith.index_cast %arg8 : i32 to index
        %12 = vector.transfer_read %10[%6, %11], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
        %13 = triton_cpu.extract_memref %3 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
        %14 = vector.transfer_read %13[%11, %7], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
        %15 = triton_cpu.dot %12, %14, %arg9, inputPrecision = tf32 : vector<32x32xbf16> * vector<32x32xbf16> -> vector<32x32xf32>
        scf.yield %15 : vector<32x32xf32>
      }
      vector.transfer_write %9, %5[%6, %7] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32, strided<[?, 1]>>
    }
  }
  tt.return
}

// -----

// AMX int8 flat/online-packing vector.contract inside of an accumulator loop

// ALL-LABEL: @gemm_amx_int8

// Tile registers for accumulation
// AMX-COUNT-4: x86.amx.tile_zero : !x86.amx.tile<16x16xi32>

// Prologue
// AMX:         scf.for %arg{{.+}} = %c0 to %c64 step %c4
// AMX:           vector.shuffle %{{.+}}, %{{.+}} [0, 32, 64, 96,
// AMX:           vector.shuffle %{{.+}}, %{{.+}} [4, 36, 68, 100,

// Main pipeline
// AMX:         %{{.+}}:4 = scf.for %arg{{.+}} = %c0 to %{{.+}} step %c64 iter_args(%arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}) ->
// AMX-SAME:      (!x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>)
// AMX-COUNT-4:   x86.amx.tile_muli

// Epilogue
// AMX:         %{{.+}}:4 = scf.for %arg{{.+}} = %{{.+}} to %{{.+}} step %c64 iter_args(%arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}) ->
// AMX-SAME:      (!x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>, !x86.amx.tile<16x16xi32>)

// Store results back to memory
// AMX-COUNT-4: x86.amx.tile_store

// Shuffle results
// AMX:         scf.for %arg{{.+}} = %c0 to %c32 step %c1

tt.func public @gemm_amx_int8(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %c0_i8 = arith.constant 0 : i8
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c32_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c32_i32 : i32
  %4 = arith.extsi %arg5 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%4, %c1_i64] : <i8>, <32x64xsi8>
  %6 = arith.extsi %arg4 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%6, %c1_i64] : <i8>, <64x32xsi8>
  %8 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%6, %c1_i64] : <i32>, <32x32xsi32>
  %9 = triton_cpu.extract_memref %8 : <32x32xsi32> -> memref<?x?xi32, strided<[?, 1]>>
  %10 = arith.index_cast %1 : i32 to index
  %11 = arith.index_cast %3 : i32 to index
  %12 = vector.transfer_read %9[%10, %11], %c0_i32 {in_bounds = [true, true]} : memref<?x?xi32, strided<[?, 1]>>, vector<32x32xi32>
  %13 = scf.for %arg6 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg7 = %12) -> (vector<32x32xi32>)  : i32 {
    %14 = triton_cpu.extract_memref %5 : <32x64xsi8> -> memref<?x?xi8, strided<[?, 1]>>
    %15 = arith.index_cast %arg6 : i32 to index
    %16 = vector.transfer_read %14[%10, %15], %c0_i8 {in_bounds = [true, true]} : memref<?x?xi8, strided<[?, 1]>>, vector<32x64xi8>
    %17 = triton_cpu.extract_memref %7 : <64x32xsi8> -> memref<?x?xi8, strided<[?, 1]>>
    %18 = vector.transfer_read %17[%15, %11], %c0_i8 {in_bounds = [true, true]} : memref<?x?xi8, strided<[?, 1]>>, vector<64x32xi8>
    %19 = triton_cpu.dot %16, %18, %arg7, inputPrecision = tf32 : vector<32x64xi8> * vector<64x32xi8> -> vector<32x32xi32>
    scf.yield %19 : vector<32x32xi32>
  }
  vector.transfer_write %13, %9[%10, %11] {in_bounds = [true, true]} : vector<32x32xi32>, memref<?x?xi32, strided<[?, 1]>>
  tt.return
}

// -----

// AVX-512 bf16 flat/online-packing vector.contract inside of an accumulator loop

// ALL-LABEL: @gemm_avx512_bf16

// Shuffle accumulator init values
// AVX512-COUNT-16:  vector.shuffle

// Main loop (using 16 accumulators (1x16xf32))
// AVX512:           %{{.+}}:16 = scf.for %arg{{.+}} = %c0 to %{{.+}} step %c2

// AVX512-COUNT-4:     vector.shuffle
// AVX512-COUNT-16:    x86.avx512.dot

// AVX512:             scf.yield

// Shuffle back before storing to memory
// AVX512-COUNT-16: vector.shuffle

tt.func public @gemm_avx512_bf16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c4_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c64_i32 : i32
  %4 = arith.extsi %arg5 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%4, %c1_i64] : <bf16>, <4x2xbf16>
  %6 = arith.extsi %arg4 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%6, %c1_i64] : <bf16>, <2x64xbf16>
  %8 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%6, %c1_i64] : <f32>, <4x64xf32>
  %9 = triton_cpu.extract_memref %8 : <4x64xf32> -> memref<?x?xf32, strided<[?, 1]>>
  %10 = arith.index_cast %1 : i32 to index
  %11 = arith.index_cast %3 : i32 to index
  %12 = vector.transfer_read %9[%10, %11], %cst_0 {in_bounds = [true, true], acc_read} : memref<?x?xf32, strided<[?, 1]>>, vector<4x64xf32>
  %13 = scf.for %arg6 = %c0_i32 to %arg5 step %c2_i32 iter_args(%arg7 = %12) -> (vector<4x64xf32>)  : i32 {
    %14 = triton_cpu.extract_memref %5 : <4x2xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
    %15 = arith.index_cast %arg6 : i32 to index
    %16 = vector.transfer_read %14[%10, %15], %cst {in_bounds = [true, true], lhs_read} : memref<?x?xbf16, strided<[?, 1]>>, vector<4x2xbf16>
    %17 = triton_cpu.extract_memref %7 : <2x64xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
    %18 = vector.transfer_read %17[%15, %11], %cst {in_bounds = [true, true], rhs_read} : memref<?x?xbf16, strided<[?, 1]>>, vector<2x64xbf16>
    %19 = triton_cpu.dot %16, %18, %arg7, inputPrecision = tf32 : vector<4x2xbf16> * vector<2x64xbf16> -> vector<4x64xf32>
    scf.yield %19 : vector<4x64xf32>
  }
  vector.transfer_write %13, %9[%10, %11] {in_bounds = [true, true], acc_write} : vector<4x64xf32>, memref<?x?xf32, strided<[?, 1]>>
  tt.return
}

// -----

// AVX10.2 int8 flat/online-packing vector.contract inside of an accumulator loop

// ALL-LABEL: @gemm_avx10_2_int8

// Shuffle accumulator init values
// AVX10_2-COUNT-16:  vector.shuffle

// Main loop (using 16 accumulators (1x16xi32))
// AVX10_2:           %{{.+}}:16 = scf.for %arg{{.+}} = %c0 to %{{.+}} step %c4

// AVX10_2-COUNT-4:     vector.shuffle
// AVX10_2-COUNT-16:    x86.avx10.dot.i8

// AVX10_2:             scf.yield

// Shuffle back before storing to memory
// AVX10_2-COUNT-16: vector.shuffle

tt.func public @gemm_avx10_2_int8(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c4_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c64_i32 : i32
  %4 = arith.extsi %arg5 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%4, %c1_i64] : <i8>, <4x4xsi8>
  %6 = arith.extsi %arg4 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%6, %c1_i64] : <i8>, <4x64xsi8>
  %8 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%6, %c1_i64] : <i32>, <4x64xsi32>
  %9 = triton_cpu.extract_memref %8 : <4x64xsi32> -> memref<?x?xi32, strided<[?, 1]>>
  %10 = arith.index_cast %1 : i32 to index
  %11 = arith.index_cast %3 : i32 to index
  %12 = vector.transfer_read %9[%10, %11], %c0_i32 {in_bounds = [true, true]} : memref<?x?xi32, strided<[?, 1]>>, vector<4x64xi32>
  %13 = scf.for %arg6 = %c0_i32 to %arg5 step %c4_i32 iter_args(%arg7 = %12) -> (vector<4x64xi32>)  : i32 {
    %14 = triton_cpu.extract_memref %5 : <4x4xsi8> -> memref<?x?xi8, strided<[?, 1]>>
    %15 = arith.index_cast %arg6 : i32 to index
    %16 = vector.transfer_read %14[%10, %15], %c0_i8 {in_bounds = [true, true]} : memref<?x?xi8, strided<[?, 1]>>, vector<4x4xi8>
    %17 = triton_cpu.extract_memref %7 : <4x64xsi8> -> memref<?x?xi8, strided<[?, 1]>>
    %18 = vector.transfer_read %17[%15, %11], %c0_i8 {in_bounds = [true, true]} : memref<?x?xi8, strided<[?, 1]>>, vector<4x64xi8>
    %19 = triton_cpu.dot %16, %18, %arg7, inputPrecision = tf32 : vector<4x4xi8> * vector<4x64xi8> -> vector<4x64xi32>
    scf.yield %19 : vector<4x64xi32>
  }
  vector.transfer_write %13, %9[%10, %11] {in_bounds = [true, true]} : vector<4x64xi32>, memref<?x?xi32, strided<[?, 1]>>
  tt.return
}

// -----

// AVX_NE_CONVERT bf16 flat/online-packing vector.contract inside of an accumulator loop

// ALL-LABEL: @gemm_avxneconvert_bf16

// Shuffle accumulator init values
// AVX_NE_CONVERT-COUNT-8:  vector.shuffle

// Main loop (using 8 accumulators (1x8xf32))
// AVX_NE_CONVERT:           %{{.+}}:8 = scf.for %arg{{.+}} = %c0 to %{{.+}} step %c1

// Code sequence of interest
// AVX_NE_CONVERT:             x86.avx.bcst_to_f32.packed
// AVX_NE_CONVERT:             x86.avx.cvt.packed.even.indexed_to_f32
// AVX_NE_CONVERT:             vector.fma
// AVX_NE_CONVERT:             x86.avx.cvt.packed.odd.indexed_to_f32
// AVX_NE_CONVERT:             vector.fma

// AVX_NE_CONVERT:             scf.yield

// Shuffle back before storing to memory
// AVX_NE_CONVERT-COUNT-8: vector.shuffle

tt.func public @gemm_avxneconvert_bf16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c2_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c32_i32 : i32
  %4 = arith.extsi %arg5 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%4, %c1_i64] : <bf16>, <2x1xbf16>
  %6 = arith.extsi %arg4 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%6, %c1_i64] : <bf16>, <1x32xbf16>
  %8 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%6, %c1_i64] : <f32>, <2x32xf32>
  %9 = triton_cpu.extract_memref %8 : <2x32xf32> -> memref<?x?xf32, strided<[?, 1]>>
  %10 = arith.index_cast %1 : i32 to index
  %11 = arith.index_cast %3 : i32 to index
  %12 = vector.transfer_read %9[%10, %11], %cst_0 {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1]>>, vector<2x32xf32>
  %13 = scf.for %arg6 = %c0_i32 to %arg5 step %c1_i32 iter_args(%arg7 = %12) -> (vector<2x32xf32>)  : i32 {
    %14 = triton_cpu.extract_memref %5 : <2x1xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
    %15 = arith.index_cast %arg6 : i32 to index
    %16 = vector.transfer_read %14[%10, %15], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<2x1xbf16>
    %17 = triton_cpu.extract_memref %7 : <1x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
    %18 = vector.transfer_read %17[%15, %11], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<1x32xbf16>
    %19 = triton_cpu.dot %16, %18, %arg7, inputPrecision = tf32 : vector<2x1xbf16> * vector<1x32xbf16> -> vector<2x32xf32>
    scf.yield %19 : vector<2x32xf32>
  }
  vector.transfer_write %13, %9[%10, %11] {in_bounds = [true, true]} : vector<2x32xf32>, memref<?x?xf32, strided<[?, 1]>>
  tt.return
}

// -----

// Temporary buffer needed: accumulator is initialized from constant

// ALL-LABEL: @gemm_amx_bf16_const_init

// AMX:         %[[ZEROVEC:.+]] = arith.constant dense<0.000000e+00> : vector<32x32xf32>
// AMX:         %[[ORIG_MEMREF:.+]] = triton_cpu.extract_memref %{{.+}} : <32x32xf32> -> memref<?x?xf32, strided<[?, 1]>

// AMX:         %[[BUFFER:.+]] = memref.alloca() : memref<32x32xf32>
// AMX:         vector.transfer_write %[[ZEROVEC]], %[[BUFFER]][%c0, %c0]

// AMX:         %[[BUFFER2:.+]] = memref.alloca() : memref<32x32xf32>
// AMX-COUNT-4: x86.amx.tile_store %[[BUFFER2]]

// AMX:         scf.for %arg{{.+}} = %c0 to %c32 step %c1
// AMX-COUNT-2:   vector.load %[[BUFFER2]]
// AMX-COUNT-2:   vector.load %[[BUFFER]]
// AMX-COUNT-2:   vector.store %{{.+}}, %[[BUFFER2]]

// AMX-COUNT-3: vector.transfer_read %[[BUFFER2]]
// AMX:         %[[LAST_READ:.+]] = vector.transfer_read %[[BUFFER2]][%c16, %c16]
// AMX-COUNT-3: vector.insert_strided_slice
// AMX:         %[[REMAT:.+]] = vector.insert_strided_slice %[[LAST_READ]], %{{.+}}
// AMX:         vector.transfer_write %[[REMAT]], %[[ORIG_MEMREF]]

tt.func public @gemm_amx_bf16_const_init(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = arith.extsi %arg5 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] : <bf16>, <32x32xbf16>
  %2 = arith.extsi %arg4 : i32 to i64
  %3 = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%2, %c1_i64] : <bf16>, <32x32xbf16>
  %4 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%2, %c1_i64] : <f32>, <32x32xf32>
  scf.for %arg6 = %c0_i32 to %arg3 step %c32_i32  : i32 {
    scf.for %arg7 = %c0_i32 to %arg4 step %c32_i32  : i32 {
      %5 = triton_cpu.extract_memref %4 : <32x32xf32> -> memref<?x?xf32, strided<[?, 1]>>
      %6 = arith.index_cast %arg6 : i32 to index
      %7 = arith.index_cast %arg7 : i32 to index
      %8 = arith.constant dense<0.0> : vector<32x32xf32>
      %9 = scf.for %arg8 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg9 = %8) -> (vector<32x32xf32>)  : i32 {
        %10 = triton_cpu.extract_memref %1 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
        %11 = arith.index_cast %arg8 : i32 to index
        %12 = vector.transfer_read %10[%6, %11], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
        %13 = triton_cpu.extract_memref %3 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
        %14 = vector.transfer_read %13[%11, %7], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
        %15 = triton_cpu.dot %12, %14, %arg9, inputPrecision = tf32 : vector<32x32xbf16> * vector<32x32xbf16> -> vector<32x32xf32>
        scf.yield %15 : vector<32x32xf32>
      }
      vector.transfer_write %9, %5[%6, %7] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32, strided<[?, 1]>>
    }
  }
  tt.return
}

// -----

// Post op

// ALL-LABEL: @gemm_amx_bf16_post_op

// AMX:         %[[ORIG_MEMREF:.+]] = triton_cpu.extract_memref %{{.+}} : <32x32xf32> -> memref<?x?xf32, strided<[?, 1]>

// If pattern applied we'll have two loops with 4 tile_mulf ops each
// AMX-COUNT-8: x86.amx.tile_mulf

// AMX:         %[[POSTOP:.+]] = math.exp %{{.+}} : vector<32x32xf32>
// AMX:         vector.transfer_write %[[POSTOP]], %[[ORIG_MEMREF]]

tt.func public @gemm_amx_bf16_post_op(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = arith.extsi %arg5 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] : <bf16>, <32x32xbf16>
  %2 = arith.extsi %arg4 : i32 to i64
  %3 = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%2, %c1_i64] : <bf16>, <32x32xbf16>
  %4 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%2, %c1_i64] : <f32>, <32x32xf32>
  scf.for %arg6 = %c0_i32 to %arg3 step %c32_i32  : i32 {
    scf.for %arg7 = %c0_i32 to %arg4 step %c32_i32  : i32 {
      %5 = triton_cpu.extract_memref %4 : <32x32xf32> -> memref<?x?xf32, strided<[?, 1]>>
      %6 = arith.index_cast %arg6 : i32 to index
      %7 = arith.index_cast %arg7 : i32 to index
      %8 = vector.transfer_read %5[%6, %7], %cst_0 {in_bounds = [true, true]} : memref<?x?xf32, strided<[?, 1]>>, vector<32x32xf32>
      %9 = scf.for %arg8 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg9 = %8) -> (vector<32x32xf32>)  : i32 {
        %10 = triton_cpu.extract_memref %1 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
        %11 = arith.index_cast %arg8 : i32 to index
        %12 = vector.transfer_read %10[%6, %11], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
        %13 = triton_cpu.extract_memref %3 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
        %14 = vector.transfer_read %13[%11, %7], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
        %15 = triton_cpu.dot %12, %14, %arg9, inputPrecision = tf32 : vector<32x32xbf16> * vector<32x32xbf16> -> vector<32x32xf32>
        scf.yield %15 : vector<32x32xf32>
      }
      %10 = math.exp %9 : vector<32x32xf32>
      vector.transfer_write %10, %5[%6, %7] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32, strided<[?, 1]>>
    }
  }
  tt.return
}


// -----

// Check application on a blocked layout

// ALL-LABEL: @gemm_amx_bf16_blocked

// Operands
// AMX:        %[[A_DESC:.+]] = tt.make_tensor_descriptor %arg0, [%{{.+}}, %{{.+}}, %c32_i32, %c32_i32], [%{{.+}}, %c32_i64, %{{.+}}, %c1_i64] : <bf16>, <1x1x32x32xbf16>
// AMX:        %[[B_DESC:.+]] = tt.make_tensor_descriptor %arg1, [%{{.+}}, %{{.+}}, %c32_i32, %c32_i32], [%{{.+}}, %c1024_i64, %c32_i64, %c1_i64] : <bf16>, <1x1x32x32xbf16>
// AMX:        %[[A_MEMREF:.+]] = triton_cpu.extract_memref %[[A_DESC]] : <1x1x32x32xbf16> -> memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>
// AMX:        %[[B_MEMREF:.+]] = triton_cpu.extract_memref %[[B_DESC]] : <1x1x32x32xbf16> -> memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>>

// Tile registers for accumulation
// AMX-COUNT-4: x86.amx.tile_zero : !x86.amx.tile<16x16xf32>

// Subview for shuffling the first B block
// AMX:         memref.subview %[[B_MEMREF]][%{{.+}}, %{{.+}}, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>> to memref<32x32xbf16, strided<[32, 1], offset: ?>>

// Prologue
// AMX:         scf.for %arg{{.+}} = %c0 to %c32 step %c2
// AMX:           vector.shuffle %{{.+}}, %{{.+}} [0, 32, 1, 33,
// AMX:           vector.shuffle %{{.+}}, %{{.+}} [4, 36, 5, 37,

// Main pipeline
// AMX:         %{{.+}}:4 = scf.for %[[K_ITER:arg.+]] = %{{.+}} to %{{.+}} step %c1 iter_args(%arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}) ->
// AMX-SAME:      (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>)

// Subview for reading the current A block
// AMX:           memref.subview %[[A_MEMREF]][%{{.+}}, %[[K_ITER]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>> to memref<32x32xbf16, strided<[?, 1], offset: ?>>

// Subview for reading the next B block
// AMX:           %[[NEXT_K:.+]] = arith.addi %[[K_ITER]], %c1 : index
// AMX:           memref.subview %[[B_MEMREF]][%{{.+}}, %[[NEXT_K]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>> to memref<32x32xbf16, strided<[32, 1], offset: ?>>

// Main computation
// AMX-COUNT-4:   x86.amx.tile_mulf

// Epilogue
// AMX:         %{{.+}}:4 = scf.for %arg{{.+}} = %{{.+}} to %{{.+}} step %c1 iter_args(%arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}, %arg{{.+}} = %{{.+}}) ->
// AMX-SAME:      (!x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>, !x86.amx.tile<16x16xf32>)

// Store results back to memory
// AMX-COUNT-4: x86.amx.tile_store

// Shuffle results
// AMX:         scf.for %arg{{.+}} = %c0 to %c32 step %c1

tt.func public @gemm_amx_bf16_blocked(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<i32>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %c32_i32 = arith.constant 32 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i64 = arith.constant 32 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i32 = arith.constant 0 : i32
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x1x32x32xbf16>
  %0 = arith.divsi %arg4, %c32_i32 : i32
  %1 = arith.divsi %arg5, %c32_i32 : i32
  %2 = arith.divsi %arg6, %c32_i32 : i32
  %3 = tt.get_program_id x : i32
  %4 = arith.muli %3, %c2_i32 : i32
  %5 = tt.addptr %arg3, %4 : !tt.ptr<i32>, i32
  %6 = tt.load %5 : !tt.ptr<i32>
  %7 = tt.addptr %5, %c1_i32 : !tt.ptr<i32>, i32
  %8 = tt.load %7 : !tt.ptr<i32>
  %9 = arith.muli %arg7, %2 : i32
  %10 = arith.muli %arg6, %c32_i32 : i32
  %11 = arith.extsi %10 : i32 to i64
  %12 = arith.extsi %arg6 : i32 to i64
  %13 = tt.make_tensor_descriptor %arg0, [%0, %2, %c32_i32, %c32_i32], [%11, %c32_i64, %12, %c1_i64] : <bf16>, <1x1x32x32xbf16>
  %14 = tt.make_tensor_descriptor %arg1, [%1, %2, %c32_i32, %c32_i32], [%11, %c1024_i64, %c32_i64, %c1_i64] : <bf16>, <1x1x32x32xbf16>
  %15 = arith.muli %arg5, %c32_i32 : i32
  %16 = arith.extsi %15 : i32 to i64
  %17 = arith.extsi %arg5 : i32 to i64
  %18 = tt.make_tensor_descriptor %arg2, [%0, %1, %c32_i32, %c32_i32], [%16, %c32_i64, %17, %c1_i64] : <bf16>, <1x1x32x32xbf16>
  %19 = arith.cmpi eq, %arg7, %c0_i32 : i32
  scf.if %19 {
    %29 = triton_cpu.extract_memref %18 : <1x1x32x32xbf16> -> memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>
    %30 = arith.index_cast %6 : i32 to index
    %31 = arith.index_cast %8 : i32 to index
    vector.transfer_write %cst_0, %29[%30, %31, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x32x32xbf16>, memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>
  }
  %20 = triton_cpu.extract_memref %18 : <1x1x32x32xbf16> -> memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>
  %21 = arith.index_cast %6 : i32 to index
  %22 = arith.index_cast %8 : i32 to index
  %23 = vector.transfer_read %20[%21, %22, %c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>, vector<32x32xbf16>
  %24 = arith.extf %23 : vector<32x32xbf16> to vector<32x32xf32>
  %25 = arith.addi %9, %2 : i32
  %26 = scf.for %arg8 = %9 to %25 step %c1_i32 iter_args(%arg9 = %24) -> (vector<32x32xf32>)  : i32 {
    %29 = triton_cpu.extract_memref %13 : <1x1x32x32xbf16> -> memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>
    %30 = arith.index_cast %arg8 : i32 to index
    %31 = vector.transfer_read %29[%21, %30, %c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>, vector<32x32xbf16>
    %32 = triton_cpu.extract_memref %14 : <1x1x32x32xbf16> -> memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>>
    %33 = vector.transfer_read %32[%22, %30, %c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?x32x32xbf16, strided<[?, 1024, 32, 1]>>, vector<32x32xbf16>
    %34 = triton_cpu.dot %31, %33, %arg9, inputPrecision = tf32 : vector<32x32xbf16> * vector<32x32xbf16> -> vector<32x32xf32>
    scf.yield %34 : vector<32x32xf32>
  }
  %27 = arith.truncf %26 : vector<32x32xf32> to vector<32x32xbf16>
  %28 = vector.shape_cast %27 : vector<32x32xbf16> to vector<1x1x32x32xbf16>
  vector.transfer_write %28, %20[%21, %22, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x32x32xbf16>, memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>
  tt.return
}
