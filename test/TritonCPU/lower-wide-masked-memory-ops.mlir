// RUN: triton-opt %s -split-input-file -triton-cpu-lower-wide-masked-memory-ops | FileCheck %s
// RUN: triton-opt %s -split-input-file -triton-cpu-lower-wide-masked-memory-ops -convert-scf-to-cf | FileCheck %s --check-prefix=CF
// RUN: triton-opt %s -split-input-file -triton-cpu-lower-wide-masked-memory-ops -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm | FileCheck %s --check-prefix=LLVM

// Wide masked load: unmasked fast path, otherwise a loop of native-width
// masked loads through function-entry stack buffers.

// CHECK-LABEL: @wide_masked_load
// CHECK-DAG:     %[[BUF:.+]] = memref.alloca() : memref<1x64xf32>
// CHECK-DAG:     %[[MBUF:.+]] = memref.alloca() : memref<64xi8>
// CHECK:         %[[ALL:.+]] = vector.reduction <and>, %{{.+}} : vector<64xi1> into i1
// CHECK:         %[[RES:.+]] = scf.if %[[ALL]] -> (vector<64xf32>) {
// CHECK:           %[[FULL:.+]] = vector.load %{{.+}}[%{{.+}}] : memref<64xf32>, vector<64xf32>
// CHECK:           scf.yield %[[FULL]]
// CHECK:         } else {
// CHECK:           vector.store %{{.+}}, %[[BUF]][%{{.+}}, %{{.+}}] : memref<1x64xf32>, vector<64xf32>
// CHECK:           vector.store %{{.+}}, %[[MBUF]][%{{.+}}] : memref<64xi8>, vector<64xi8>
// CHECK:           scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK:             %[[M:.+]] = arith.cmpi ne, %{{.+}}, %{{.+}} : vector<4xi8>
// CHECK:             %[[PT:.+]] = vector.load %[[BUF]][%{{.+}}, %[[IV]]] : memref<1x64xf32>, vector<4xf32>
// CHECK:             %[[V:.+]] = vector.maskedload %{{.+}}[%{{.+}}], %[[M]], %[[PT]] : memref<64xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:             vector.store %[[V]], %[[BUF]][%{{.+}}, %[[IV]]] : memref<1x64xf32>, vector<4xf32>
// CHECK:           vector.load %[[BUF]][%{{.+}}, %{{.+}}] : memref<1x64xf32>, vector<64xf32>
// CHECK:         tt.return %[[RES]]

// CF-LABEL: @wide_masked_load
// CF:         memref.alloca() : memref<1x64xf32>
// CF:         cf.cond_br
// CF:         vector.maskedload {{.+}} into vector<4xf32>
// CF:         tt.return

tt.func @wide_masked_load(%base: memref<64xf32>, %idx: index, %mask: vector<64xi1>, %pass: vector<64xf32>) -> vector<64xf32> {
  %0 = vector.maskedload %base[%idx], %mask, %pass : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  tt.return %0 : vector<64xf32>
}

// -----

// CHECK-LABEL: @wide_masked_store
// CHECK-DAG:     memref.alloca() : memref<1x64xf32>
// CHECK-DAG:     memref.alloca() : memref<64xi8>
// CHECK:         %[[ALL:.+]] = vector.reduction <and>, %{{.+}} : vector<64xi1> into i1
// CHECK:         scf.if %[[ALL]] {
// CHECK:           vector.store %{{.+}}, %{{.+}}[%{{.+}}] : memref<64xf32>, vector<64xf32>
// CHECK:         } else {
// CHECK:           scf.for
// CHECK:             vector.maskedstore %{{.+}}[%{{.+}}], %{{.+}}, %{{.+}} : memref<64xf32>, vector<4xi1>, vector<4xf32>
// CHECK-NOT:     vector.maskedstore {{.+}} vector<64xf32>

tt.func @wide_masked_store(%base: memref<64xf32>, %idx: index, %mask: vector<64xi1>, %val: vector<64xf32>) {
  vector.maskedstore %base[%idx], %mask, %val : memref<64xf32>, vector<64xi1>, vector<64xf32>
  tt.return
}

// -----

// Inside a loop the buffers still go in the entry block, so the stack does
// not grow per iteration.

// CHECK-LABEL: @wide_masked_store_in_loop
// CHECK-DAG:     memref.alloca() : memref<1x64xf32>
// CHECK-DAG:     memref.alloca() : memref<64xi8>
// CHECK:         scf.for
// CHECK-NOT:       memref.alloca
// CHECK:           vector.reduction <and>

tt.func @wide_masked_store_in_loop(%base: memref<64xf32>, %n: index, %mask: vector<64xi1>, %val: vector<64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    vector.maskedstore %base[%c0], %mask, %val : memref<64xf32>, vector<64xi1>, vector<64xf32>
  }
  tt.return
}

// -----

// Two wide loads give two fast-path blocks and two buffer reloads. They must
// not look identical, or convert-vector-to-llvm merges them while memrefs are
// still unlowered and passes a memref to an llvm.cond_br.

// LLVM-LABEL: @two_wide_masked_loads
// LLVM-NOT:     ^bb{{[0-9]+}}({{[^)]*}}memref
// LLVM:         tt.return

tt.func @two_wide_masked_loads(%a: memref<64xf32>, %b: memref<64xf32>, %idx: index, %mask: vector<64xi1>, %pass: vector<64xf32>) -> vector<64xf32> {
  %0 = vector.maskedload %a[%idx], %mask, %pass : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  %1 = vector.maskedload %b[%idx], %mask, %pass : memref<64xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  %2 = arith.addf %0, %1 : vector<64xf32>
  tt.return %2 : vector<64xf32>
}

// -----

// Ops that already fit one register are left alone.

// CHECK-LABEL: @native_masked_load
// CHECK-NOT:     scf.if
// CHECK:         vector.maskedload {{.+}} into vector<4xf32>

tt.func @native_masked_load(%base: memref<64xf32>, %idx: index, %mask: vector<4xi1>, %pass: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.maskedload %base[%idx], %mask, %pass : memref<64xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  tt.return %0 : vector<4xf32>
}

// -----

// Wide gather: the index vector is spilled once and reloaded one register per
// iteration, so LLVM expands the per-lane gather inside a loop body.

// CHECK-LABEL: @wide_gather
// CHECK-DAG:     %[[BUF:.+]] = memref.alloca() : memref<64xf32>
// CHECK-DAG:     %[[IBUF:.+]] = memref.alloca() : memref<64xi64>
// CHECK-DAG:     %[[MBUF:.+]] = memref.alloca() : memref<64xi8>
// CHECK:         vector.store %{{.+}}, %[[BUF]][%{{.+}}] : memref<64xf32>, vector<64xf32>
// CHECK:         vector.store %{{.+}}, %[[IBUF]][%{{.+}}] : memref<64xi64>, vector<64xi64>
// CHECK:         vector.store %{{.+}}, %[[MBUF]][%{{.+}}] : memref<64xi8>, vector<64xi8>
// CHECK:         scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK:           %[[M:.+]] = arith.cmpi ne, %{{.+}}, %{{.+}} : vector<4xi8>
// CHECK:           %[[IDX:.+]] = vector.load %[[IBUF]][%[[IV]]] : memref<64xi64>, vector<4xi64>
// CHECK:           %[[PT:.+]] = vector.load %[[BUF]]
// CHECK:           %[[V:.+]] = vector.gather %{{.+}}[] [%[[IDX]]], %[[M]], %[[PT]] : memref<f32>, vector<4xi64>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:           vector.store %[[V]], %[[BUF]]
// CHECK:         }
// CHECK:         %[[RES:.+]] = vector.load %[[BUF]]
// CHECK:         tt.return %[[RES]]
// CHECK-NOT:     vector.gather {{.+}} vector<64xf32>

tt.func @wide_gather(%base: memref<f32>, %idx: vector<64xi64>, %mask: vector<64xi1>, %pass: vector<64xf32>) -> vector<64xf32> {
  %0 = vector.gather %base[] [%idx], %mask, %pass : memref<f32>, vector<64xi64>, vector<64xi1>, vector<64xf32> into vector<64xf32>
  tt.return %0 : vector<64xf32>
}

// -----

// Wide scatter: value, indices and mask are spilled once and replayed in
// register-sized chunks.

// CHECK-LABEL: @wide_scatter
// CHECK-DAG:     %[[BUF:.+]] = memref.alloca() : memref<1x64xf32>
// CHECK-DAG:     %[[IBUF:.+]] = memref.alloca() : memref<64xi64>
// CHECK-DAG:     %[[MBUF:.+]] = memref.alloca() : memref<64xi8>
// CHECK:         scf.for %[[IV:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK:           %[[M:.+]] = arith.cmpi ne, %{{.+}}, %{{.+}} : vector<4xi8>
// CHECK:           %[[IDX:.+]] = vector.load %[[IBUF]][%[[IV]]] : memref<64xi64>, vector<4xi64>
// CHECK:           %[[VAL:.+]] = vector.load %[[BUF]]
// CHECK:           vector.scatter %{{.+}}[%{{.+}}] [%[[IDX]]], %[[M]], %[[VAL]] : memref<64xf32>, vector<4xi64>, vector<4xi1>, vector<4xf32>
// CHECK:         }
// CHECK-NOT:     vector.scatter {{.+}} vector<64xf32>

tt.func @wide_scatter(%base: memref<64xf32>, %idx: vector<64xi64>, %mask: vector<64xi1>, %val: vector<64xf32>) {
  %c0 = arith.constant 0 : index
  vector.scatter %base[%c0] [%idx], %mask, %val : memref<64xf32>, vector<64xi64>, vector<64xi1>, vector<64xf32>
  tt.return
}

// -----

// Gathers that already fit one register are left alone.

// CHECK-LABEL: @native_gather
// CHECK-NOT:     scf.for
// CHECK:         vector.gather {{.+}} into vector<4xf32>

tt.func @native_gather(%base: memref<f32>, %idx: vector<4xi64>, %mask: vector<4xi1>, %pass: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.gather %base[] [%idx], %mask, %pass : memref<f32>, vector<4xi64>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  tt.return %0 : vector<4xf32>
}
