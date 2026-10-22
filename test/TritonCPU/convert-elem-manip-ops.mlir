// RUN: triton-opt %s -split-input-file -triton-cpu-convert-elem-manip-ops -canonicalize | FileCheck %s --implicit-check-not=tt.gather --implicit-check-not=memref.load --implicit-check-not=memref.store --implicit-check-not=scf.for

// Gather through the middle axis, with a larger result and 64-bit indices.
// Materialize only the source and retain vector indices in the result shape.
// CHECK-LABEL: @gather_middle_axis
// CHECK-DAG: %[[POISON:.+]] = ub.poison : vector<2x16x8xf32>
// CHECK-DAG: %[[ALLTRUE:.+]] = arith.constant dense<true> : vector<2x16x8xi1>
// CHECK: %[[SOURCE:.+]] = memref.alloca() {{.*}} : memref<2x4x8xf32>
// CHECK: vector.transfer_write {{.*}}, %[[SOURCE]][{{.*}}]
// CHECK: %[[FLAT:.+]] = memref.collapse_shape %[[SOURCE]] {{.*}} : memref<2x4x8xf32> into memref<64xf32>
// CHECK: %[[MASK:.+]] = arith.cmpi ult, {{.*}} : vector<2x16x8xi64>
// CHECK: %[[SAFE:.+]] = arith.select %[[MASK]], {{.*}} : vector<2x16x8xi1>, vector<2x16x8xi64>
// CHECK: %[[SCALED:.+]] = arith.muli %[[SAFE]], {{.*}} : vector<2x16x8xi64>
// CHECK: %[[INDICES:.+]] = arith.addi %[[SCALED]], {{.*}} : vector<2x16x8xi64>
// CHECK: vector.gather %[[FLAT]][{{.*}}] [%[[INDICES]]], %[[ALLTRUE]], %[[POISON]] : memref<64xf32>, vector<2x16x8xi64>, vector<2x16x8xi1>, vector<2x16x8xf32> into vector<2x16x8xf32>
tt.func @gather_middle_axis(%src: tensor<2x4x8xf32>, %indices: tensor<2x16x8xi64>) -> tensor<2x16x8xf32> {
  %result = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<2x4x8xf32>, tensor<2x16x8xi64>) -> tensor<2x16x8xf32>
  tt.return %result : tensor<2x16x8xf32>
}

// -----

// Narrow indices must be zero-extended: i8 values 128..255 can be valid indices.
// Sub-byte sources need byte-addressed scratch for standard vector.gather.
// CHECK-LABEL: @gather_unsigned_indices
// CHECK-DAG: %[[BYTES:.+]] = arith.extui {{.*}} : vector<256xi4> to vector<256xi8>
// CHECK-DAG: %[[BUFFER:.+]] = memref.alloca() {{.*}} : memref<256xi8>
// CHECK: vector.transfer_write %[[BYTES]], %[[BUFFER]][{{.*}}]
// CHECK: arith.extui {{.*}} : vector<64xi8> to vector<64xi64>
// CHECK: vector.gather %[[BUFFER]][{{.*}}] {{.*}} : memref<256xi8>, vector<64xi64>, vector<64xi1>, vector<64xi8> into vector<64xi8>
// CHECK: arith.trunci {{.*}} : vector<64xi8> to vector<64xi4>
tt.func @gather_unsigned_indices(%src: tensor<256xi4>, %indices: tensor<64xi8>) -> tensor<64xi4> {
  %result = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<256xi4>, tensor<64xi8>) -> tensor<64xi4>
  tt.return %result : tensor<64xi4>
}

// -----

// Fixed scratch allocations must precede the first branch, even when the
// source loop is already control flow and has no enclosing SCF operation.
// CHECK-LABEL: @gather_in_loop
// CHECK: memref.alloca() {{.*}} : memref<4xf32>
// CHECK-NOT: memref.alloca
// CHECK: cf.br ^[[HEADER:bb[0-9]+]]
// CHECK-NOT: memref.alloca
// CHECK: ^[[HEADER]](
// CHECK-NOT: memref.alloca
// CHECK: cf.cond_br {{.*}}, ^[[BODY:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
// CHECK-NOT: memref.alloca
// CHECK: ^[[BODY]]:
// CHECK-NOT: memref.alloca
// CHECK: vector.transfer_write
// CHECK-NOT: memref.alloca
// CHECK-NOT: cf.br
// CHECK-NOT: cf.cond_br
// CHECK: vector.gather
// CHECK-NOT: memref.alloca
// CHECK-NOT: cf.cond_br
// CHECK: cf.br ^[[HEADER]]
// CHECK-NOT: memref.alloca
// CHECK: ^[[EXIT]]:
// CHECK-NOT: memref.alloca
// CHECK: tt.return
tt.func @gather_in_loop(%src: tensor<4xf32>, %indices: tensor<4xi32>, %out: tensor<4x!tt.ptr<f32>>, %count: i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.br ^loop(%zero : i32)
^loop(%i: i32):
  %continue = arith.cmpi slt, %i, %count : i32
  cf.cond_br %continue, ^body, ^exit
^body:
  %result = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  tt.store %out, %result : tensor<4x!tt.ptr<f32>>
  %next = arith.addi %i, %one : i32
  cf.br ^loop(%next : i32)
^exit:
  tt.return
}
