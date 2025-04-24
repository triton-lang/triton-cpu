// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=sse" | FileCheck %s --check-prefix=CHECK-SSE
// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=sse,sse2,sse3" | FileCheck %s --check-prefix=CHECK-SSE
// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=avx" | FileCheck %s --check-prefix=CHECK-AVX
// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=avx,avx2" | FileCheck %s --check-prefix=CHECK-AVX
// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=avx,sse" | FileCheck %s --check-prefix=CHECK-AVX
// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=avx512f" | FileCheck %s --check-prefix=CHECK-AVX512F
// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=avx512f,avx" | FileCheck %s --check-prefix=CHECK-AVX512F
// RUN: triton-opt %s -split-input-file -triton-cpu-math-to-vec-lib="cpu_features=avx512f,avx,sse" | FileCheck %s --check-prefix=CHECK-AVX512F

// Convert math ops to VecLib ops.

// CHECK-SSE-LABEL: @exp_kernel
// CHECK-SSE: %[[EXTRACTED:.*]] = vector.extract %{{.*}}[0] : vector<4xf32> from vector<256x4xf32>
// CHECK-SSE-NEXT: %[[CALLED:.*]] = func.call @Sleef_expf4_u10(%[[EXTRACTED]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-SSE-NEXT: %[[INSERTED:.*]] = vector.insert %[[CALLED]], %{{.*}}[0] : vector<4xf32> into vector<256x4xf32>

// CHECK-AVX-LABEL: @exp_kernel
// CHECK-AVX: %[[EXTRACTED:.*]] = vector.extract %{{.*}}[0] : vector<8xf32> from vector<128x8xf32>
// CHECK-AVX-NEXT: %[[CALLED:.*]] = func.call @Sleef_expf8_u10(%[[EXTRACTED]]) : (vector<8xf32>) -> vector<8xf32>
// CHECK-AVX-NEXT: %[[INSERTED:.*]] = vector.insert %[[CALLED]], %{{.*}}[0] : vector<8xf32> into vector<128x8xf32>

// CHECK-AVX512F-LABEL: @exp_kernel
// CHECK-AVX512F: %[[EXTRACTED:.*]] = vector.extract %{{.*}}[0] : vector<16xf32> from vector<64x16xf32>
// CHECK-AVX512F-NEXT: %[[CALLED:.*]] = func.call @Sleef_expf16_u10(%[[EXTRACTED]]) : (vector<16xf32>) -> vector<16xf32>
// CHECK-AVX512F-NEXT: %[[INSERTED:.*]] = vector.insert %[[CALLED]], %{{.*}}[0] : vector<16xf32> into vector<64x16xf32>

module {
  tt.func public @exp_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg2: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = triton_cpu.ptr_to_memref %2 : <f32> -> memref<1024xf32>
    %4 = vector.load %3[%c0] : memref<1024xf32>, vector<1024xf32>
    %5 = math.exp %4 : vector<1024xf32>
    %6 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %7 = triton_cpu.ptr_to_memref %6 : <f32> -> memref<1024xf32>
    vector.store %5, %7[%c0] : memref<1024xf32>, vector<1024xf32>
    tt.return
  }
}
