#include "cpu/include/Dialect/TritonCPU/IR/Types.h"
#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/TypeSwitch.h"           // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::cpu;

#define GET_TYPEDEF_CLASSES
#include "cpu/include/Dialect/TritonCPU/IR/Types.cpp.inc"

Type triton::cpu::TokenType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  int type = 1;
  if (parser.parseInteger(type))
    return Type();

  if (parser.parseGreater())
    return Type();

  return triton::cpu::TokenType::get(parser.getContext(), type);
}

void triton::cpu::TokenType::print(AsmPrinter &printer) const {
  printer << "<" << getType() << ">";
}

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::cpu::TritonCPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cpu/include/Dialect/TritonCPU/IR/Types.cpp.inc"
      >();
}
