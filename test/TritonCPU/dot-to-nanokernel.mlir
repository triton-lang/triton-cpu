// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=amx-tile -cse  | FileCheck %s --check-prefixes=AMX,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avx512f -cse  | FileCheck %s --check-prefixes=AVX512,ALL

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
