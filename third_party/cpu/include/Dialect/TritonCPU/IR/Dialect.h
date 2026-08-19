#ifndef TRITON_DIALECT_TRITONCPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONCPU_IR_DIALECT_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonCPU depends on Triton
#include "cpu/include/Dialect/TritonCPU/IR/Attributes.h"
#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h.inc"
#include "cpu/include/Dialect/TritonCPU/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GET_OP_CLASSES
#include "cpu/include/Dialect/TritonCPU/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TRITONCPU_IR_DIALECT_H_
