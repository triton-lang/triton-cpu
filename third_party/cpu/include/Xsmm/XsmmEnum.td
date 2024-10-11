//===- XsmmEnum --------------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def Xsmm_DataType: I64EnumAttr<
    "DataType", "see: libxsmm_datatype",
    [
      I64EnumAttrCase<"F32",  1, "f32">,
      I64EnumAttrCase<"BF16", 2, "bf16">
    ]>{
   let cppNamespace = "mlir::xsmm";
}

def Xsmm_BinaryKind : I64EnumAttr<
    "BinaryKind", "see: libxsmm_meltw_binary_type",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"ADD", 1, "add">,
      I64EnumAttrCase<"MUL", 2, "mul">,
      I64EnumAttrCase<"SUB", 3, "sub">,
      I64EnumAttrCase<"DIV", 4, "div">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_UnaryKind : I64EnumAttr<
    "UnaryKind", "see: libxsmm_meltw_unary_type",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"IDENTITY", 1, "identity">,
      I64EnumAttrCase<"ZERO", 2, "zero">,
      I64EnumAttrCase<"RELU", 5, "relu">,
      I64EnumAttrCase<"VNNI2", 28, "vnni_2">,
      I64EnumAttrCase<"TRANSPOSE", 29, "transpose">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_UnaryFlags : I64EnumAttr<
    "UnaryFlags", "see: libxsmm_meltw_unary_flags",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"BCAST_ROW", 2, "bcast_row">,
      I64EnumAttrCase<"BCAST_COL", 4, "bcast_col">,
      I64EnumAttrCase<"BCAST_SCALAR", 8, "bcast_scalar">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_BinaryFlags : I64EnumAttr<
    "BinaryFlags", "see: libxsmm_meltw_binary_flags",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"BCAST_ROW_IN_0", 1, "bcast_row_in0">,
      I64EnumAttrCase<"BCAST_ROW_IN_1", 2, "bcast_row_in1">,
      I64EnumAttrCase<"BCAST_COL_IN_0", 4, "bcast_col_in0">,
      I64EnumAttrCase<"BCAST_COL_IN_1", 8, "bcast_col_in1">,
      I64EnumAttrCase<"BCAST_SCALAR_IN_0", 16, "bcast_scalar_in0">,
      I64EnumAttrCase<"BCAST_SCALAR_IN_1", 32, "bcast_scalar_in1">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_GemmFlags : I64EnumAttr<
    "GemmFlags", "see: libxsmm_gemm_flags",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"BETA_0", 4, "beta_0">,
      I64EnumAttrCase<"VNNI_A", 2048, "vnni_a">,
      I64EnumAttrCase<"VNNI_B", 4096, "vnni_b">,
      I64EnumAttrCase<"VNNI_C", 8192, "vnni_c">,
      I64EnumAttrCase<"NO_RESET_TILECONFIG", 64, "no_reset_tileconfig">,
      I64EnumAttrCase<"NO_SETUP_TILECONFIG", 128, "no_setup_tileconfig">
   ]> {
  let cppNamespace = "mlir::xsmm";
}
