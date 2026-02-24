#ifndef TRITONCPU_CONVERSION_TRITONCPUOPT_OPTCOMMON_H
#define TRITONCPU_CONVERSION_TRITONCPUOPT_OPTCOMMON_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace triton {
namespace cpu {

inline Type getElemTyOrTy(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.getElementType();
  return ty;
}

inline bool isTyOrVectorOf(Type ty, Type elemTy) {
  return getElemTyOrTy(ty) == elemTy;
}

inline bool isBf16(Type ty) {
  return isTyOrVectorOf(ty, BFloat16Type::get(ty.getContext()));
}

inline bool isFp16(Type ty) {
  return isTyOrVectorOf(ty, Float16Type::get(ty.getContext()));
}

inline bool isFp32(Type ty) {
  return isTyOrVectorOf(ty, Float32Type::get(ty.getContext()));
}

inline bool isFp8(Type ty) {
  Type elemTy = getElemTyOrTy(ty);
  if (elemTy.isIntOrFloat() && !elemTy.isInteger())
    return elemTy.getIntOrFloatBitWidth() == 8;
  return false;
}

inline Type toTyOrVectorOf(Type ty, Type elemTy) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.cloneWith(std::nullopt, elemTy);
  return elemTy;
}

inline Type toInt8(Type ty) {
  return toTyOrVectorOf(ty, IntegerType::get(ty.getContext(), 8));
}

inline Type toInt16(Type ty) {
  return toTyOrVectorOf(ty, IntegerType::get(ty.getContext(), 16));
}

inline Type toInt32(Type ty) {
  return toTyOrVectorOf(ty, IntegerType::get(ty.getContext(), 32));
}

inline Type toInt64(Type ty) {
  return toTyOrVectorOf(ty, IntegerType::get(ty.getContext(), 64));
}

inline Type toFp8E5M2(Type ty) {
  return toTyOrVectorOf(ty, Float8E5M2Type::get(ty.getContext()));
}

inline Type toFp16(Type ty) {
  return toTyOrVectorOf(ty, Float16Type::get(ty.getContext()));
}

inline Type toBf16(Type ty) {
  return toTyOrVectorOf(ty, BFloat16Type::get(ty.getContext()));
}

inline Type toFp32(Type ty) {
  return toTyOrVectorOf(ty, Float32Type::get(ty.getContext()));
}

inline Value intCst(Location loc, Type ty, int64_t val,
                    PatternRewriter &rewriter) {
  TypedAttr valAttr = IntegerAttr::get(getElemTyOrTy(ty), val);
  if (auto vecTy = dyn_cast<VectorType>(ty))
    valAttr = SplatElementsAttr::get(vecTy, valAttr);
  return arith::ConstantOp::create(rewriter, loc, valAttr);
}

inline Value fpCst(Location loc, Type ty, double val,
                   PatternRewriter &rewriter) {
  TypedAttr valAttr = FloatAttr::get(getElemTyOrTy(ty), val);
  if (auto vecTy = dyn_cast<VectorType>(ty))
    valAttr = SplatElementsAttr::get(vecTy, valAttr);
  return arith::ConstantOp::create(rewriter, loc, valAttr);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
Value cstLike(Location loc, Value tySrc, T val, PatternRewriter &rewriter) {
  return intCst(loc, tySrc.getType(), val, rewriter);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
Value cstLike(Location loc, Value tySrc, T val, PatternRewriter &rewriter) {
  return fpCst(loc, tySrc.getType(), val, rewriter);
}

inline Value shapeCast(Location loc, Value in, VectorType outTy,
                       PatternRewriter &rewriter) {
  VectorType inTy = cast<VectorType>(in.getType());
  assert(outTy.getElementType() == inTy.getElementType());
  assert(outTy.getNumElements() == inTy.getNumElements());
  return vector::ShapeCastOp::create(rewriter, loc, outTy, in);
}

inline Value shapeCast(Location loc, Value in,
                       std::initializer_list<int64_t> shapes,
                       PatternRewriter &rewriter) {
  VectorType inTy = cast<VectorType>(in.getType());
  VectorType outTy = VectorType::get(shapes, inTy.getElementType());
  return shapeCast(loc, in, outTy, rewriter);
}

} // namespace cpu
} // namespace triton
} // namespace mlir

#define int_cst(ty, val) intCst(loc, ty, val, rewriter)
#define index_cst(val) arith::ConstantIndexOp::create(rewriter, loc, val)
#define cst_like(src, val) cstLike(loc, src, val, rewriter)

#define op_addi(lhs, rhs) arith::AddIOp::create(rewriter, loc, lhs, rhs)
#define op_addf(lhs, rhs) arith::AddFOp::create(rewriter, loc, lhs, rhs)
#define op_subi(lhs, rhs) arith::SubIOp::create(rewriter, loc, lhs, rhs)
#define op_subf(lhs, rhs) arith::SubFOp::create(rewriter, loc, lhs, rhs)
#define op_muli(lhs, rhs) arith::MulIOp::create(rewriter, loc, lhs, rhs)
#define op_mulf(lhs, rhs) arith::MulFOp::create(rewriter, loc, lhs, rhs)
#define op_divsi(lhs, rhs) arith::DivSIOp::create(rewriter, loc, lhs, rhs)
#define op_divui(lhs, rhs) arith::DivUIOp::create(rewriter, loc, lhs, rhs)
#define op_bitcast(ty, val) arith::BitcastOp::create(rewriter, loc, ty, val)
#define op_lshr(lhs, rhs) arith::ShRUIOp::create(rewriter, loc, lhs, rhs)
#define op_shl(lhs, rhs) arith::ShLIOp::create(rewriter, loc, lhs, rhs)
#define op_trunci(ty, val) arith::TruncIOp::create(rewriter, loc, ty, val)
#define op_zext(ty, val) arith::ExtUIOp::create(rewriter, loc, ty, val)
#define op_sext(ty, val) arith::ExtSIOp::create(rewriter, loc, ty, val)
#define op_and(lhs, rhs) arith::AndIOp::create(rewriter, loc, lhs, rhs)
#define op_or(lhs, rhs) arith::OrIOp::create(rewriter, loc, lhs, rhs)
#define op_minui(lhs, rhs) arith::MinUIOp::create(rewriter, loc, lhs, rhs)
#define op_maxui(lhs, rhs) arith::MaxUIOp::create(rewriter, loc, lhs, rhs)
#define op_select(cond, val, other)                                            \
  arith::SelectOp::create(rewriter, loc, cond, val, other)
#define op_sitofp(ty, val) arith::SIToFPOp::create(rewriter, loc, ty, val)
#define op_fptosi(ty, val) arith::FPToSIOp::create(rewriter, loc, ty, val)
#define op_read(ty, memRef, indices)                                           \
  vector::TransferReadOp::create(                                              \
      rewriter, loc, ty, memRef, indices,                                      \
      arith::getZeroConstant(rewriter, loc, ty.getElementType()),              \
      SmallVector<bool>(ty.getRank(), true))
#define op_write(val, memRef, indices)                                         \
  vector::TransferWriteOp::create(                                             \
      rewriter, loc, val, memRef, indices,                                     \
      SmallVector<bool>(cast<VectorType>(val.getType()).getRank(), true))
#define op_interleave(lhs, rhs)                                                \
  vector::InterleaveOp::create(rewriter, loc, lhs, rhs)
#define op_extract(vec, idx) vector::ExtractOp::create(rewriter, loc, vec, idx)
#define op_store(val, mem, idx)                                                \
  vector::StoreOp::create(rewriter, loc, val, mem, idx)
#define op_index_cast(ty, val)                                                 \
  arith::IndexCastOp::create(rewriter, loc, ty, val)
#define op_icmp_eq(lhs, rhs)                                                   \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, lhs, rhs)
#define op_icmp_ne(lhs, rhs)                                                   \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, lhs, rhs)
#define op_icmp_ugt(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt, lhs, rhs)
#define op_icmp_uge(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge, lhs, rhs)
#define op_icmp_ult(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult, lhs, rhs)
#define op_icmp_ule(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ule, lhs, rhs)
#define op_icmp_sgt(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt, lhs, rhs)
#define op_icmp_sge(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, lhs, rhs)
#define op_icmp_slt(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, lhs, rhs)
#define op_icmp_sle(lhs, rhs)                                                  \
  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle, lhs, rhs)

#endif
