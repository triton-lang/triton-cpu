// RUN: triton-opt %s -split-input-file -triton-cpu-decompose-fp-conversions="decompose-bf16-conversions=true decompose-fp8-conversions=true" | FileCheck %s

// fp32 -> bf16 rounds to nearest even by default and keeps NaNs quiet.

// CHECK-LABEL: @fp32_to_bf16_rne
// CHECK-DAG:     %[[C16:.+]] = arith.constant dense<16> : vector<8xi32>
// CHECK-DAG:     %[[C1:.+]] = arith.constant dense<1> : vector<8xi32>
// CHECK-DAG:     %[[BIAS:.+]] = arith.constant dense<32767> : vector<8xi32>
// CHECK:         %[[SRC:.+]] = arith.bitcast %{{.+}} : vector<8xf32> to vector<8xi32>
// CHECK:         %[[HI:.+]] = arith.shrui %[[SRC]], %[[C16]]
// CHECK:         %[[LSB:.+]] = arith.andi %[[HI]], %[[C1]]
// CHECK:         %[[ADD:.+]] = arith.addi %[[LSB]], %[[BIAS]]
// CHECK:         %[[ROUNDED:.+]] = arith.addi %[[SRC]], %[[ADD]]
// CHECK:         %[[ISNAN:.+]] = arith.cmpi ugt
// CHECK:         %[[QNAN:.+]] = arith.ori %[[SRC]]
// CHECK:         %[[SEL:.+]] = arith.select %[[ISNAN]], %[[QNAN]], %[[ROUNDED]]
// CHECK:         arith.shrui %[[SEL]]
// CHECK:         arith.trunci {{.+}} : vector<8xi32> to vector<8xi16>
// CHECK-NOT:     arith.truncf
tt.func @fp32_to_bf16_rne(%arg0: vector<8xf32>) -> vector<8xbf16> {
  %0 = arith.truncf %arg0 : vector<8xf32> to vector<8xbf16>
  tt.return %0 : vector<8xbf16>
}

// -----

// Round toward zero keeps plain truncation.

// CHECK-LABEL: @fp32_to_bf16_rtz
// CHECK:         %[[C16:.+]] = arith.constant dense<16> : vector<8xi32>
// CHECK-NEXT:    %[[SRC:.+]] = arith.bitcast %{{.+}} : vector<8xf32> to vector<8xi32>
// CHECK-NEXT:    %[[HI:.+]] = arith.shrui %[[SRC]], %[[C16]]
// CHECK-NEXT:    arith.trunci %[[HI]] : vector<8xi32> to vector<8xi16>
// CHECK-NOT:     arith.truncf
tt.func @fp32_to_bf16_rtz(%arg0: vector<8xf32>) -> vector<8xbf16> {
  %0 = arith.truncf %arg0 toward_zero : vector<8xf32> to vector<8xbf16>
  tt.return %0 : vector<8xbf16>
}

// -----

// CHECK-LABEL: @e4m3fn_to_fp16
// CHECK-DAG:   %[[MASK16:.+]] = arith.constant 127 : i16
// CHECK-DAG:   %[[NAN16:.+]] = arith.constant 0x7E00 : f16
// CHECK:       %[[SRC16:.+]] = arith.bitcast %{{.+}} : f8E4M3FN to i8
// CHECK:       %[[BITS16:.+]] = arith.extui %[[SRC16]] : i8 to i16
// CHECK:       arith.mulf
// CHECK:       %[[MAG16:.+]] = arith.andi %[[BITS16]], %[[MASK16]] : i16
// CHECK:       %[[ISNAN16:.+]] = arith.cmpi eq, %[[MAG16]], %[[MASK16]] : i16
// CHECK:       %[[RESULT16:.+]] = arith.select %[[ISNAN16]], %[[NAN16]], %{{.+}} : f16
// CHECK:       tt.return %[[RESULT16]] : f16
tt.func @e4m3fn_to_fp16(%arg0: f8E4M3FN) -> f16 {
  %0 = arith.extf %arg0 : f8E4M3FN to f16
  tt.return %0 : f16
}

// -----

// CHECK-LABEL: @e4m3fn_to_fp32
// CHECK-DAG:   %[[MASK32:.+]] = arith.constant dense<127> : vector<8xi32>
// CHECK-DAG:   %[[NAN32:.+]] = arith.constant dense<0x7FC00000> : vector<8xf32>
// CHECK:       %[[SRC32:.+]] = arith.bitcast %{{.+}} : vector<8xf8E4M3FN> to vector<8xi8>
// CHECK:       %[[BITS32:.+]] = arith.extui %[[SRC32]] : vector<8xi8> to vector<8xi32>
// CHECK:       arith.mulf
// CHECK:       %[[MAG32:.+]] = arith.andi %[[BITS32]], %[[MASK32]] : vector<8xi32>
// CHECK:       %[[ISNAN32:.+]] = arith.cmpi eq, %[[MAG32]], %[[MASK32]] : vector<8xi32>
// CHECK:       %[[RESULT32:.+]] = arith.select %[[ISNAN32]], %[[NAN32]], %{{.+}} : vector<8xi1>, vector<8xf32>
// CHECK:       tt.return %[[RESULT32]] : vector<8xf32>
tt.func @e4m3fn_to_fp32(%arg0: vector<8xf8E4M3FN>) -> vector<8xf32> {
  %0 = arith.extf %arg0 : vector<8xf8E4M3FN> to vector<8xf32>
  tt.return %0 : vector<8xf32>
}
