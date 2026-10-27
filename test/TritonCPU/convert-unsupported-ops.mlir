// RUN: triton-opt %s -split-input-file -triton-cpu-add-casts-for-unsupported-ops="promote-bf16-to-fp32=true" | FileCheck %s

// BF16 gathers and scatters only move bits, so they are done on i16.

// CHECK-LABEL: @bf16_gather
// CHECK:         %[[BASE:.+]] = triton_cpu.ptr_to_memref %{{.+}} : <bf16> -> memref<1xi16>
// CHECK:         %[[PASS:.+]] = arith.bitcast %{{.+}} : vector<8xbf16> to vector<8xi16>
// CHECK:         %[[GATHER:.+]] = vector.gather %[[BASE]][%{{.+}}] [%{{.+}}], %{{.+}}, %[[PASS]] : memref<1xi16>, vector<8xi32>, vector<8xi1>, vector<8xi16> into vector<8xi16>
// CHECK:         arith.bitcast %[[GATHER]] : vector<8xi16> to vector<8xbf16>
tt.func @bf16_gather(%ptr: !tt.ptr<bf16>, %idx: vector<8xi32>, %mask: vector<8xi1>, %pass: vector<8xbf16>) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %base = triton_cpu.ptr_to_memref %ptr : <bf16> -> memref<1xbf16>
  %res = vector.gather %base[%c0] [%idx], %mask, %pass : memref<1xbf16>, vector<8xi32>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
  tt.return %res : vector<8xbf16>
}

// -----

// CHECK-LABEL: @bf16_scatter
// CHECK:         %[[BASE:.+]] = triton_cpu.ptr_to_memref %{{.+}} : <bf16> -> memref<1xi16>
// CHECK:         %[[VAL:.+]] = arith.bitcast %{{.+}} : vector<8xbf16> to vector<8xi16>
// CHECK:         vector.scatter %[[BASE]][%{{.+}}] [%{{.+}}], %{{.+}}, %[[VAL]] : memref<1xi16>, vector<8xi32>, vector<8xi1>, vector<8xi16>
tt.func @bf16_scatter(%ptr: !tt.ptr<bf16>, %idx: vector<8xi32>, %mask: vector<8xi1>, %val: vector<8xbf16>) {
  %c0 = arith.constant 0 : index
  %base = triton_cpu.ptr_to_memref %ptr : <bf16> -> memref<1xbf16>
  vector.scatter %base[%c0] [%idx], %mask, %val : memref<1xbf16>, vector<8xi32>, vector<8xi1>, vector<8xbf16>
  tt.return
}

// -----

// BF16 compares are done in FP32.

// CHECK-LABEL: @bf16_cmpf
// CHECK:         %[[LHS:.+]] = arith.extf %{{.+}} : vector<8xbf16> to vector<8xf32>
// CHECK:         %[[RHS:.+]] = arith.extf %{{.+}} : vector<8xbf16> to vector<8xf32>
// CHECK:         arith.cmpf oge, %[[LHS]], %[[RHS]] : vector<8xf32>
tt.func @bf16_cmpf(%a: vector<8xbf16>, %b: vector<8xbf16>) -> vector<8xi1> {
  %res = arith.cmpf oge, %a, %b : vector<8xbf16>
  tt.return %res : vector<8xi1>
}
