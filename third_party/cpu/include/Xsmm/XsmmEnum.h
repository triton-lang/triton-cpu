//===- XsmmEnum.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_XSMM_XSMMENUM_H
#define TPP_DIALECT_XSMM_XSMMENUM_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "cpu/include/Xsmm/XsmmEnum.h.inc"

#endif // TPP_DIALECT_XSMM_XSMMENUM_H
