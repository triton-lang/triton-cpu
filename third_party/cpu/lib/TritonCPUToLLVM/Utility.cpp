#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::cpu {

Value getProgramId(mlir::FunctionOpInterface funcOp, int axis) {
  auto args = funcOp.getArguments();
  assert(funcOp && args.size() >= kNumProgramContextArgs);
  assert(axis >= 0 && axis < static_cast<int>(kNumProgramDims));

  // The first three of the last six args are x, y, z program ids.
  auto argIdx = args.size() - kNumProgramContextArgs + axis;
  assert(argIdx < args.size() && "out-of-bounds arg index");
  assert(args[argIdx].getType().isInteger(32) && "unexpected arg type");
  return args[argIdx];
}

Value getNumPrograms(mlir::FunctionOpInterface funcOp, int axis) {
  auto args = funcOp.getArguments();
  assert(funcOp && args.size() >= kNumProgramContextArgs);
  assert(axis >= 0 && axis < static_cast<int>(kNumProgramDims));

  // The last three of the args are gridX, gridY, gridZ (bounds) of grid.
  auto argIdx = args.size() - kNumProgramDims + axis;
  assert(argIdx < args.size() && "out-of-bounds arg index");
  assert(args[argIdx].getType().isInteger(32) && "unexpected arg type");
  return args[argIdx];
}

} // namespace mlir::triton::cpu
