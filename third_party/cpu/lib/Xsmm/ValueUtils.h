//===- ValueUtils.h - -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_VALUEUTILS_H
#define TPP_TRANSFORMS_UTILS_VALUEUTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <utility>

using namespace mlir;

namespace mlir {
class OpBuilder;
class Operation;
class Location;
namespace utils {

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val);

// Returns true if the op defining `val` represents a zero filled tensor.
bool isZeroTensor(Value val);

// Returns true if the operation represents a zero filled tensor.
bool isZeroOp(Operation *);

// Returns the strides of `val`. The method returns something usefull
// only if the `val` type is a strided memref and the strides are statically
// known.
FailureOr<SmallVector<int64_t>> getStaticStrides(Value val);

// Return the offset and ptr for `val`. Assert if `val`
// is not a memref.
std::pair<Value, Value> getPtrAndOffset(OpBuilder &builder, Value val,
                                        Location loc);

} // namespace utils
} // namespace mlir

#endif // TPP_TRANSFORMS_UTILS_VALUEUTILS_H
