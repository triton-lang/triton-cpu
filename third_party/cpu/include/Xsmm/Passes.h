#ifndef TritonCPUXsmm_CONVERSION_PASSES_H
#define TritonCPUXsmm_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;

namespace affine {
class AffineDialect;
} // namespace affine

namespace arith {
class ArithDialect;
} // namespace arith

namespace func {
class FuncOp;
class FuncDialect;
} // namespace func

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace math {
class MathDialect;
} // namespace math

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace vector {
class VectorDialect;
} // namespace vector

} // namespace mlir

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/Xsmm/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "cpu/include/Xsmm/Passes.h.inc"

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif
