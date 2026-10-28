// RUN: triton-opt %s -split-input-file -triton-cpu-convert-elem-manip-ops -triton-cpu-map-elementwise-to-llvm -triton-cpu-convert-elem-manip-ops -canonicalize | FileCheck %s --implicit-check-not=tt.map_elementwise --implicit-check-not=tt.gather

// Nested maps must be lowered after inlining, including in bodies that
// return early from different control-flow blocks.
// CHECK-LABEL: @map_conditional_returns
// CHECK: llvm.extractelement
// CHECK: llvm.cond_br
// CHECK: llvm.br
// CHECK: llvm.insertelement
tt.func @map_conditional_returns(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = "tt.map_elementwise"(%input) <{pack = 1 : i32}> ({
  ^bb0(%x: i32):
    %zero = arith.constant 0 : i32
    %tiny = vector.broadcast %x : i32 to vector<1xi32>
    %tiny_tensor = builtin.unrealized_conversion_cast %tiny : vector<1xi32> to tensor<1xi32>
    %nested = "tt.map_elementwise"(%tiny_tensor) <{pack = 1 : i32}> ({
    ^bb0(%inner: i32):
      %twice = arith.addi %inner, %inner : i32
      tt.map_elementwise.return %twice : i32
    }) : (tensor<1xi32>) -> tensor<1xi32>
    %nested_vector = builtin.unrealized_conversion_cast %nested : tensor<1xi32> to vector<1xi32>
    %mapped = vector.extract %nested_vector[0] : i32 from vector<1xi32>
    %negative = arith.cmpi slt, %mapped, %zero : i32
    cf.cond_br %negative, ^bb1, ^bb2
  ^bb1:
    %negated = arith.subi %zero, %mapped : i32
    tt.map_elementwise.return %negated : i32
  ^bb2:
    tt.map_elementwise.return %mapped : i32
  }) : (tensor<8xi32>) -> tensor<8xi32>
  tt.return %result : tensor<8xi32>
}

// -----

// Packs are grouped by operand and results retain their individual types.
// Gather scratch writes must be deferred until the map body is inlined.
// CHECK-LABEL: @map_packed_mixed_outputs
// CHECK: llvm.extractvalue
// CHECK: llvm.extractelement {{.*}} : vector<4xi32>
// CHECK: llvm.extractelement {{.*}} : vector<4xf32>
// CHECK: vector.gather
// CHECK: llvm.insertelement {{.*}} : vector<4xi32>
// CHECK: llvm.insertelement {{.*}} : vector<4xf32>
// CHECK: llvm.insertvalue
tt.func @map_packed_mixed_outputs(%input: tensor<2x4xi32>, %other: tensor<2x4xf32>) -> (tensor<2x4xi32>, tensor<2x4xf32>) {
  %result:2 = "tt.map_elementwise"(%input, %other) <{pack = 2 : i32}> ({
  ^bb0(%a0: i32, %a1: i32, %b0: f32, %b1: f32):
    %pair = vector.from_elements %a0, %a1 : vector<2xi32>
    %pair_tensor = builtin.unrealized_conversion_cast %pair : vector<2xi32> to tensor<2xi32>
    %indices = arith.constant dense<[1, 0]> : vector<2xi32>
    %indices_tensor = builtin.unrealized_conversion_cast %indices : vector<2xi32> to tensor<2xi32>
    %gathered = tt.gather %pair_tensor[%indices_tensor] {axis = 0 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %gathered_vector = builtin.unrealized_conversion_cast %gathered : tensor<2xi32> to vector<2xi32>
    %lhs = vector.extract %gathered_vector[0] : i32 from vector<2xi32>
    %rhs = vector.extract %gathered_vector[1] : i32 from vector<2xi32>
    %sum = arith.addi %lhs, %rhs : i32
    %difference = arith.subf %b0, %b1 : f32
    tt.map_elementwise.return %sum, %a0, %difference, %b0 : i32, i32, f32, f32
  }) : (tensor<2x4xi32>, tensor<2x4xf32>) -> (tensor<2x4xi32>, tensor<2x4xf32>)
  tt.return %result#0, %result#1 : tensor<2x4xi32>, tensor<2x4xf32>
}
