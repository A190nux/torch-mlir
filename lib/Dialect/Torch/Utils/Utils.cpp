//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/SparsityUtils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

int64_t Torch::toPositiveDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 ? dim : dim + inputRank;
}

bool Torch::isValidDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 && dim < inputRank;
}

std::optional<int64_t>
Torch::matchLegalConstantIndexIntoListOfSize(Value v, int64_t length) {
  int64_t dim;
  if (!matchPattern(v, m_TorchConstantInt(&dim)))
    return std::nullopt;
  dim = toPositiveDim(dim, length);
  if (!isValidDim(dim, length))
    return std::nullopt;
  return dim;
}

Value Torch::toIntListConstruct(PatternRewriter &rewriter, Location loc,
                                ArrayRef<int64_t> cstInput) {
  SmallVector<Value> cstValues;
  for (int64_t i : cstInput) {
    cstValues.push_back(rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(i)));
  }
  return rewriter.create<Torch::PrimListConstructOp>(
      loc, Torch::ListType::get(IntType::get(rewriter.getContext())),
      cstValues);
}

bool Torch::getListConstructElements(Value v, SmallVectorImpl<Value> &elems) {
  auto listConstruct = v.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return false;
  elems = llvm::to_vector<4>(listConstruct.getElements());
  return true;
}

torch_upstream::ScalarType Torch::getScalarTypeForType(Type type) {
  if (isa<Float32Type>(type))
    return torch_upstream::ScalarType::Float;
  if (isa<Float64Type>(type))
    return torch_upstream::ScalarType::Double;
  if (type.isSignedInteger(64))
    return torch_upstream::ScalarType::Long;
  if (type.isSignedInteger(32))
    return torch_upstream::ScalarType::Int;
  if (type.isSignedInteger(16))
    return torch_upstream::ScalarType::Short;
  if (type.isSignlessInteger(1))
    return torch_upstream::ScalarType::Bool;
  if (type.isBF16())
    return torch_upstream::ScalarType::BFloat16;
  if (type.isF16())
    return torch_upstream::ScalarType::Half;
  if (type.isUnsignedInteger(8))
    return torch_upstream::ScalarType::Byte;
  if (type.isSignedInteger(8))
    return torch_upstream::ScalarType::Char;
  if (isa<QUInt8Type>(type))
    return torch_upstream::ScalarType::QUInt8;
  if (isa<QInt8Type>(type))
    return torch_upstream::ScalarType::QInt8;
  if (isa<QInt16Type>(type))
    return torch_upstream::ScalarType::QInt16;
  if (isa<QInt32Type>(type))
    return torch_upstream::ScalarType::QInt32;
  if (isa<ComplexType>(type)) {
    mlir::Type complexElemType = cast<ComplexType>(type).getElementType();
    if (complexElemType.isF16())
      return torch_upstream::ScalarType::ComplexHalf;
    if (complexElemType.isF32())
      return torch_upstream::ScalarType::ComplexFloat;
    if (complexElemType.isF64())
      return torch_upstream::ScalarType::ComplexDouble;
  }
  if (isa<Float8E5M2Type>(type))
    return torch_upstream::ScalarType::Float8_e5m2;
  if (isa<Float8E4M3FNType>(type))
    return torch_upstream::ScalarType::Float8_e4m3fn;
  if (isa<Float8E5M2FNUZType>(type))
    return torch_upstream::ScalarType::Float8_e5m2fnuz;
  if (isa<Float8E4M3FNUZType>(type))
    return torch_upstream::ScalarType::Float8_e4m3fnuz;
  std::string errorMsg = "Unhandled type in getScalarTypeForType: ";
  llvm::raw_string_ostream os(errorMsg);
  type.print(os);
  // os << "\nType ID: " << type.getTypeID();
  os << "\nType properties:";
  os << "\n  Is integer: " << (type.isInteger() ? "yes" : "no");
  os << "\n  Is float: "
     << (type.isIntOrFloat() && !type.isInteger() ? "yes" : "no");
  os << "\n  Is index: " << (type.isIndex() ? "yes" : "no");
  os << "\n  Bit width: "
     << (type.isIntOrFloat() ? std::to_string(type.getIntOrFloatBitWidth())
                             : "N/A");
  os << "\n  Is signless: " << (type.isSignlessInteger() ? "yes" : "no");
  os << "\n  Is signed: " << (type.isSignedInteger() ? "yes" : "no");
  // special error message for unsigned integer
  if (type.isUnsignedInteger()) {
    os << "\n  Is unsigned: yes";
    os << "\nUnsigned integer support is currently spotty. Please seeheck "
          "https://github.com/llvm/torch-mlir/issues/3720 "
          "for more details.";
  }
  llvm::report_fatal_error(llvm::StringRef(errorMsg));
}
Type Torch::getTypeForTorchType(
    MLIRContext *context, Type type,
    mlir::IntegerType::SignednessSemantics signedness) {
  if (isa<Torch::IntType>(type))
    return IntegerType::get(context, 64, signedness);
  if (isa<Torch::FloatType>(type))
    return Float64Type::get(context);
  llvm::report_fatal_error("unhandled type for getTypeForTorchType");
}

FailureOr<Type>
Torch::getTypeForScalarType(MLIRContext *context,
                            torch_upstream::ScalarType dtypeInt) {
  switch (dtypeInt) {
  case torch_upstream::ScalarType::Float:
    return Float32Type::get(context);
  case torch_upstream::ScalarType::Double:
    return Float64Type::get(context);
  case torch_upstream::ScalarType::Long:
    return IntegerType::get(context, 64, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::Int:
    return IntegerType::get(context, 32, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::Short:
    return IntegerType::get(context, 16, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::Bool:
    return IntegerType::get(context, 1);
  case torch_upstream::ScalarType::BFloat16:
    return mlir::BFloat16Type::get(context);
  case torch_upstream::ScalarType::Half:
    return mlir::Float16Type::get(context);
  case torch_upstream::ScalarType::Byte:
    return mlir::IntegerType::get(context, 8, mlir::IntegerType::Unsigned);
  case torch_upstream::ScalarType::Char:
    return mlir::IntegerType::get(context, 8, mlir::IntegerType::Signed);
  case torch_upstream::ScalarType::QUInt8:
    return QUInt8Type::get(context);
  case torch_upstream::ScalarType::QInt8:
    return QInt8Type::get(context);
  case torch_upstream::ScalarType::QInt16:
    return QInt16Type::get(context);
  case torch_upstream::ScalarType::QInt32:
    return QInt32Type::get(context);
  case torch_upstream::ScalarType::ComplexHalf:
    return mlir::ComplexType::get(Float16Type::get(context));
  case torch_upstream::ScalarType::ComplexFloat:
    return mlir::ComplexType::get(Float32Type::get(context));
  case torch_upstream::ScalarType::ComplexDouble:
    return mlir::ComplexType::get(Float64Type::get(context));
  case torch_upstream::ScalarType::Float8_e5m2:
    return Float8E5M2Type::get(context);
  case torch_upstream::ScalarType::Float8_e4m3fn:
    return Float8E4M3FNType::get(context);
  case torch_upstream::ScalarType::Float8_e5m2fnuz:
    return Float8E5M2FNUZType::get(context);
  case torch_upstream::ScalarType::Float8_e4m3fnuz:
    return Float8E4M3FNUZType::get(context);
  case torch_upstream::ScalarType::Undefined:
    return failure();
  default:
    llvm::report_fatal_error("unhandled type for getTypeForScalarType");
  }
}

FailureOr<Type>
Torch::getTorchTypeForScalarType(MLIRContext *context,
                                 torch_upstream::ScalarType dtypeInt) {
  switch (dtypeInt) {
  case torch_upstream::ScalarType::Double:
    return Torch::FloatType::get(context);
  case torch_upstream::ScalarType::Long:
    return Torch::IntType::get(context);
  case torch_upstream::ScalarType::Undefined:
  default:
    return failure();
  }
}

Type Torch::getDefaultDtypeForTorchScalar(Type type) {
  MLIRContext *context = type.getContext();
  if (isa<Torch::FloatType>(type)) {
    // For now, use float32 which is the initial default dtype returned by
    // `torch.get_default_dtype`.
    return Float32Type::get(context);
  }
  if (isa<Torch::IntType>(type))
    return IntegerType::get(context, 64, IntegerType::Signed);
  if (isa<Torch::BoolType>(type))
    return IntegerType::get(context, 1);
  llvm_unreachable(
      "getDefaultDtypeForTorchScalar called on an unsupported type");
}

Type Torch::getBuiltInTypeForTorchScalar(Type type) {
  MLIRContext *context = type.getContext();
  if (isa<Torch::FloatType>(type))
    return Float64Type::get(context);
  if (isa<Torch::IntType>(type))
    return IntegerType::get(context, 64, IntegerType::Signed);
  if (isa<Torch::BoolType>(type))
    return IntegerType::get(context, 1);
  llvm_unreachable(
      "getBuiltInTypeForTorchScalar called on an unsupported type");
}

Value Torch::getDtypeIntValueForType(PatternRewriter &rewriter, Location loc,
                                     Type dtype) {
  int intType = (int)getScalarTypeForType(dtype);
  return rewriter.create<ConstantIntOp>(loc,
                                        rewriter.getI64IntegerAttr(intType));
}

template <typename OpTy>
FailureOr<Value> Torch::getDtypeFromOp(PatternRewriter &rewriter, OpTy op) {
  // For ops like AtenEmptyLikeOp, if dtype specified in the op is none, then it
  // defaults to dtype of input. Since dtype specifies the dtype of output, in
  // this scenario we can look at dtype of output instead of input itself.
  // For ops like AtenRandOp, if dtype specified in the op is none, then it
  // defaults to a global value. In this case as well we can look at dtype of
  // output as it will already be set according to the default global value.
  Value dtype = op.getDtype();
  if (isa<Torch::NoneType>(dtype.getType())) {
    BaseTensorType tensorType = cast<BaseTensorType>(op.getType());
    if (!tensorType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected input tensor to have a dtype");
    }
    dtype =
        getDtypeIntValueForType(rewriter, op.getLoc(), tensorType.getDtype());
  }
  return dtype;
}
// Template instantiation template std::optional<Value>
template FailureOr<Value>
Torch::getDtypeFromOp<AtenEmptyLikeOp>(PatternRewriter &rewriter,
                                       AtenEmptyLikeOp op);
template FailureOr<Value>
Torch::getDtypeFromOp<AtenNewEmptyOp>(PatternRewriter &rewriter,
                                      AtenNewEmptyOp op);
template FailureOr<Value>
Torch::getDtypeFromOp<AtenRandOp>(PatternRewriter &rewriter, AtenRandOp op);
template FailureOr<Value>
Torch::getDtypeFromOp<AtenEmptyStridedOp>(PatternRewriter &rewriter,
                                          AtenEmptyStridedOp op);
template FailureOr<Value>
Torch::getDtypeFromOp<AtenRandnGeneratorOp>(PatternRewriter &rewriter,
                                            AtenRandnGeneratorOp op);

// Helper to convert a tensor to a specific scalar type.
Value Torch::convertTensorToDtype(PatternRewriter &rewriter, Location loc,
                                  Value input, Type dtype) {
  BaseTensorType origType = cast<BaseTensorType>(input.getType());
  Type newType = origType.getWithSizesAndDtype(origType.getSizes(), dtype);
  // `convertIntVal` contains the corresponding integer for the dtype which is
  // used by the aten.to.dtype op.
  Value convertIntVal = getDtypeIntValueForType(rewriter, loc, dtype);
  Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  Value converted = rewriter.create<AtenToDtypeOp>(
      loc, newType, input, convertIntVal, falseVal, falseVal, noneVal);
  return converted;
}

bool Torch::isBuiltInType(Type type) {
  return isa<BuiltinDialect>(type.getDialect());
}

std::optional<unsigned> Torch::getTensorRank(Value tensor) {
  BaseTensorType tensorType = cast<BaseTensorType>(tensor.getType());
  if (!tensorType.hasSizes())
    return std::nullopt;
  return tensorType.getSizes().size();
}

std::optional<int64_t> Torch::getTensorNumel(Value tensor) {
  BaseTensorType tensorType = cast<BaseTensorType>(tensor.getType());
  if (!tensorType.hasSizes())
    return std::nullopt;
  int64_t numel = 1;
  for (auto dim : tensorType.getSizes()) {
    if (dim == ShapedType::kDynamic)
      return ShapedType::kDynamic;
    numel *= dim;
  }
  return numel;
}

bool Torch::isViewLikeOp(Operation *op) {
  // AtenContiguousOp might return a view, so this is conservatively
  // correct. We could potentially be more precise and identify the cases
  // that it does not return a view and treat those as having value
  // semantics.
  return isa<AtenAsStridedOp, AtenBroadcastToOp, AtenContiguousOp, AtenDetachOp,
             AtenExpandAsOp, AtenExpandOp, AtenFlattenUsingIntsOp,
             AtenUnflattenIntOp, AtenPermuteOp, AtenReshapeOp,
             Aten_ReshapeAliasOp, AtenSelectIntOp, AtenSliceTensorOp,
             AtenSqueezeDimOp, AtenSqueezeOp, AtenTOp, AtenToDtypeOp,
             AtenTransposeIntOp, AtenUnsqueezeOp, AtenViewOp,
             TensorStaticInfoCastOp, AtenToDtypeLayoutOp, AtenNumpyTOp,
             AtenNarrowOp, AtenNarrowTensorOp, AtenToDeviceOp, PrimsSqueezeOp,
             AtenMovedimIntOp, PrimsViewOfOp, AtenRealOp, AtenImagOp,
             PrimsSplitDimOp, AtenViewAsComplexOp, AtenViewAsRealOp,
             AtenPixelShuffleOp, AtenPixelUnshuffleOp, AtenDiagonalOp, AtenUnfoldOp>(op);
}

Value Torch::getConstantWithGivenDtypeAndValue(PatternRewriter &rewriter,
                                               Location loc, float value,
                                               Type dtype) {
  // Creating constants satisfying backend contract.
  if (dtype.isInteger(64) || dtype.isInteger(32) || dtype.isInteger(16) ||
      dtype.isInteger(8) || dtype.isInteger(1))
    return rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr((int64_t)value));
  if (dtype.isF64() || dtype.isF32() || dtype.isF16() || dtype.isBF16() ||
      isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
          Float8E4M3FNUZType>(dtype))
    return rewriter.create<ConstantFloatOp>(loc,
                                            rewriter.getF64FloatAttr(value));
  llvm::report_fatal_error(
      "unhandled type for getConstantWithGivenDtypeAndValue");
}

// Return the number of elements of a tensor if the shape is static; otherwise,
// return -1.
int64_t Torch::getNumberOfElements(RankedTensorType inputType) {
  if (!inputType.hasStaticShape())
    return -1;
  SmallVector<int64_t> inputShape =
      makeShapeTorchCompatible(inputType.getShape());
  int64_t numel = 1;
  for (int64_t i = 0; i < inputType.getRank(); i++)
    numel *= inputShape[i];
  return numel;
}

SmallVector<int64_t> Torch::makeShapeLLVMCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kUnknownSize);
    if (shape[i] == kUnknownSize)
      updatedShape[i] = kDynamic;
  }
  return updatedShape;
}

SmallVector<int64_t> Torch::makeShapeTorchCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kDynamic);
    if (shape[i] == kDynamic)
      updatedShape[i] = kUnknownSize;
  }
  return updatedShape;
}

ValueTensorType Torch::getTensorTypeFromShapeValues(ArrayRef<Value> shapes,
                                                    Type dtype) {
  assert(!shapes.empty() && "shape vector cannot be empty");
  SmallVector<int64_t> shapeInts;
  for (Value shape : shapes) {
    int64_t dim;
    if (matchPattern(shape, m_TorchConstantInt(&dim)))
      shapeInts.push_back(dim);
    else
      shapeInts.push_back(kUnknownSize);
  }
  return Torch::ValueTensorType::get(shapes[0].getContext(), shapeInts, dtype);
}

// Helper function to get the size of the tensor at the given dimension.
Value Torch::getTensorDimSize(PatternRewriter &rewriter, Value tensor,
                              int64_t dim) {
  auto loc = tensor.getLoc();
  auto dimVal =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(dim));
  // Use 'createOrFold' instead of 'create':
  // If the dimension is a constant, then the AtenSizeIntOp is folded to a
  // ContantIntOp.
  return rewriter.createOrFold<AtenSizeIntOp>(loc, tensor, dimVal);
}

// Helper function to squeeze the input tensor at given dim.
// Return the squeezed tensor or failure.
FailureOr<Value> Torch::squeezeTensor(PatternRewriter &rewriter, Operation *op,
                                      Location loc, int64_t dim, Value input) {
  BaseTensorType inputType = cast<BaseTensorType>(input.getType());
  if (!inputType.hasSizes()) {
    return rewriter.notifyMatchFailure(loc, "input tensor must have size");
  }
  SmallVector<int64_t> inputShape{inputType.getSizes()};
  unsigned inputRank = inputShape.size();
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank)) {
    return rewriter.notifyMatchFailure(
        op, "dimension to be squeezed is an invalid dim");
  }
  inputShape.erase(inputShape.begin() + dim);
  Type squeezedType =
      inputType.getWithSizesAndDtype(inputShape, inputType.getOptionalDtype());

  Value cstDim = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(dim));
  // Adding a check to verify if the dimension to be squeezed has size 1 or not.
  Value cstOne =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value dimSize = rewriter.create<AtenSizeIntOp>(loc, input, cstDim);
  Value cmp = rewriter.create<Torch::AtenEqIntOp>(loc, dimSize, cstOne);
  rewriter.create<Torch::RuntimeAssertOp>(
      loc, cmp,
      "squeeze operation possible for dim only when input_shape[dim] == 1.");

  Value result =
      rewriter.create<AtenSqueezeDimOp>(loc, squeezedType, input, cstDim);
  return result;
}

// Helper function to unsqueeze the input tensor at given dim.
// Return the unsqueezed tensor or failure.
FailureOr<Value> Torch::unsqueezeTensor(PatternRewriter &rewriter,
                                        Operation *op, Value input, Value dim) {
  BaseTensorType inputType = cast<BaseTensorType>(input.getType());
  if (!inputType.hasSizes()) {
    return rewriter.notifyMatchFailure(op, "input tensor must have size");
  }
  FailureOr<Attribute> enc =
      getSparsityWithDenseLTAtDim(inputType.getOptionalSparsity(), dim);
  if (failed(enc)) {
    return failure();
  }

  SmallVector<int64_t> unsqueezedShape;
  ArrayRef<int64_t> inputShape = inputType.getSizes();
  // `input` has a reduced rank. Hence add 1.
  int64_t unsqueezedRank = inputShape.size() + 1;
  int64_t dimInt = 0;
  if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
    dimInt = toPositiveDim(dimInt, unsqueezedRank);
    if (!isValidDim(dimInt, unsqueezedRank)) {
      return rewriter.notifyMatchFailure(op, "dim is not a valid dim");
    }
    unsqueezedShape.append(inputShape.begin(), inputShape.end());
    unsqueezedShape.insert(unsqueezedShape.begin() + dimInt, 1);
  } else {
    unsqueezedShape.resize(unsqueezedRank, kUnknownSize);
  }
  Type unsqueezedType = inputType.getWithSizesAndDtypeAndSparsity(
      unsqueezedShape, inputType.getOptionalDtype(), enc.value());
  Value unsqueezed = rewriter.create<AtenUnsqueezeOp>(
      op->getLoc(), unsqueezedType, input, dim);
  return unsqueezed;
}

// Checks whether the `shapeA` and `shapeB` are broadcast compatible or not. If
// yes, then computes the final broadcast shape.
void Torch::computeBroadcastShape(PatternRewriter &rewriter, Location loc,
                                  Value inputA, Value inputB,
                                  SmallVector<int64_t> &resultShape,
                                  SmallVector<Value> &resultShapeValue) {
  SmallVector<int64_t> shapeA{
      cast<BaseTensorType>(inputA.getType()).getSizes()};
  SmallVector<int64_t> shapeB{
      cast<BaseTensorType>(inputB.getType()).getSizes()};
  unsigned rankA = shapeA.size();
  unsigned rankB = shapeB.size();
  unsigned minRank = rankA > rankB ? rankB : rankA;
  // Check whether the shapes of the tensors are broadcastable or not.
  // Two tensors are “broadcastable” if the following rules hold:
  // 1.) Each tensor has at least one dimension.
  // 2.) When iterating over the dimension sizes, starting at the trailing
  // dimension, the dimension sizes must either be equal, one of them is 1, or
  // one of them does not exist.
  for (unsigned i = 0; i < minRank; i++) {
    Value sizeDimA = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankA - i - 1));
    Value sizeDimB = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankB - i - 1));
    Value sizeInputA =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputA, sizeDimA);
    Value sizeInputB =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputB, sizeDimB);
    Value torchCstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value cmpSizeAEqualsSizeB =
        rewriter.create<Torch::AtenEqIntOp>(loc, sizeInputA, sizeInputB);
    Value cmpSizeAEqualsOne =
        rewriter.create<Torch::AtenEqIntOp>(loc, sizeInputA, torchCstOne);
    Value cmpSizeBEqualsOne =
        rewriter.create<Torch::AtenEqIntOp>(loc, sizeInputB, torchCstOne);
    Value anyBoolOpList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(cmpSizeAEqualsOne.getType()),
        SmallVector<Value>{cmpSizeAEqualsSizeB, cmpSizeAEqualsOne,
                           cmpSizeBEqualsOne});
    Value cmp = rewriter.create<Torch::AtenAnyBoolOp>(loc, anyBoolOpList);
    rewriter.create<Torch::RuntimeAssertOp>(
        loc, cmp, "tensors are not broadcast compatible");
  }
  // If we reach here then it means both the shapes are broadcast compatible.
  resultShape = rankA >= rankB ? shapeA : shapeB;
  Value shapeTensor = rankA >= rankB ? inputA : inputB;
  for (unsigned i = 0; i < resultShape.size(); i++) {
    Value sizeDim = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(i));
    resultShapeValue.push_back(
        rewriter.createOrFold<AtenSizeIntOp>(loc, shapeTensor, sizeDim));
  }

  unsigned resultRank = resultShape.size();
  for (unsigned i = 0; i < minRank; i++) {
    Value sizeDimA = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankA - i - 1));
    Value sizeDimB = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rankB - i - 1));
    Value sizeInputA =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputA, sizeDimA);
    Value sizeInputB =
        rewriter.createOrFold<AtenSizeIntOp>(loc, inputB, sizeDimB);
    resultShapeValue[resultRank - i - 1] =
        rewriter.create<PrimMaxIntOp>(loc, sizeInputA, sizeInputB);
    if (shapeA[rankA - i - 1] == kUnknownSize ||
        shapeB[rankB - i - 1] == kUnknownSize) {
      resultShape[resultRank - i - 1] = kUnknownSize;
    } else {
      resultShape[resultRank - i - 1] =
          std::max(shapeA[rankA - i - 1], shapeB[rankB - i - 1]);
    }
  }
}

bool Torch::isAssumingStrictSymbolicShapes(Block *block) {
  for (Operation *parentOp = block->getParentOp(); parentOp;
       parentOp = parentOp->getParentOp()) {
    if (parentOp->hasAttr("torch.assume_strict_symbolic_shapes"))
      return true;
  }
  return false;
}

LogicalResult Torch::checkDefaultStrideHelper(Operation *op,
                                              PatternRewriter &rewriter,
                                              Value opSize, Value opStride,
                                              Location loc) {

  SmallVector<int64_t> sizeListInts, strideListInts;
  if (matchPattern(opSize, m_TorchListOfConstantInts(sizeListInts)) &&
      matchPattern(opStride, m_TorchListOfConstantInts(strideListInts))) {

    // We only support the cases with default stride values.
    // For ex: aten.new_empty_strided(self, size=[2, 3, 4], stride=[12, 4, 1])
    // Here the stride[0] == size[1] * size[2], stride[1] == size[2], and
    // stride[2] == 1.
    bool isDefaultStride = true;
    for (unsigned i = 0; i < strideListInts.size(); i++) {
      int64_t defaultStride = 1;
      for (unsigned j = i + 1; j < sizeListInts.size(); j++)
        defaultStride *= sizeListInts[j];
      if (defaultStride != strideListInts[i]) {
        isDefaultStride = false;
        break;
      }
    }
    if (!isDefaultStride)
      return rewriter.notifyMatchFailure(
          op, "only default strides supported for empty_strided op");

    return success();

  } else {
    SmallVector<Value> sizeListValues;
    if (!getListConstructElements(opSize, sizeListValues))
      return rewriter.notifyMatchFailure(op, "couldn't get size list values");
    SmallVector<Value> strideListValues;
    if (!getListConstructElements(opStride, strideListValues))
      return rewriter.notifyMatchFailure(op,
                                         "couldn't get stride list values.");
    SmallVector<Value> boolVector;
    for (unsigned i = 0; i < strideListValues.size(); i++) {
      Value defaultStride = rewriter.createOrFold<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(1));
      for (unsigned j = i + 1; j < sizeListValues.size(); j++) {
        defaultStride = rewriter.createOrFold<Torch::AtenMulIntOp>(
            loc, defaultStride, sizeListValues[j]);
      }
      boolVector.push_back(rewriter.createOrFold<Torch::AtenEqIntOp>(
          loc, defaultStride, strideListValues[i]));
    }
    Value allBoolOpList = rewriter.createOrFold<PrimListConstructOp>(
        loc, Torch::ListType::get(rewriter.getType<Torch::BoolType>()),
        boolVector);
    Value cmp = rewriter.createOrFold<Torch::AtenAllBoolOp>(loc, allBoolOpList);
    rewriter.createOrFold<Torch::RuntimeAssertOp>(
        loc, cmp, "not all strides are default");
    return success();
  }
}

// Helper to create a tensor filled with the given scalar. Scalar would be
// converted the to the element type of the given tensor type.
Value Torch::createInitTensor(PatternRewriter &rewriter, Location loc,
                              BaseTensorType resultType, Value scalar,
                              Value sizeList) {
  assert(resultType.hasDtype() && "result must have dtype");
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  Value dtype = getDtypeIntValueForType(rewriter, loc, resultType.getDtype());
  return rewriter.create<AtenFullOp>(loc, resultType, sizeList, scalar, dtype,
                                     /*layout=*/noneVal,
                                     /*device=*/noneVal,
                                     /*memory_format=*/noneVal);
}

// Helper to create a rank 0 tensor filled with the given `scalar`. `scalar`
// would be converted to the element type of the given `inputType`.
Value Torch::createRank0Tensor(PatternRewriter &rewriter, Location loc,
                               BaseTensorType inputType, Value scalar) {
  assert(inputType.hasDtype() && "input must have dtype");
  SmallVector<int64_t> sizes;
  BaseTensorType rank0TensorTy = cast<BaseTensorType>(
      inputType.getWithSizesAndDtype(ArrayRef(sizes), inputType.getDtype()));
  Value dimList = rewriter.create<PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(inputType.getContext())),
      ValueRange{});
  return createInitTensor(rewriter, loc, rank0TensorTy, scalar, dimList);
}

LogicalResult Torch::getTransposedType(BaseTensorType inType, int64_t dimA,
                                       int64_t dimB, Type &transposedType) {
  if (!inType.hasSizes())
    return failure();
  SmallVector<int64_t> shape(inType.getSizes());
  int64_t tmp = shape[dimA];
  shape[dimA] = shape[dimB];
  shape[dimB] = tmp;
  transposedType = inType.getWithSizesAndDtype(llvm::ArrayRef(shape),
                                               inType.getOptionalDtype());
  return success();
}

LogicalResult Torch::getPermutedType(BaseTensorType inType,
                                     SmallVector<int64_t> permuteDims,
                                     Type &permutedType) {
  if (!inType.hasSizes())
    return failure();

  SmallVector<int64_t> shape(inType.getSizes());
  if (shape.size() != permuteDims.size())
    return failure();

  SmallVector<int64_t> permutedShape;
  for (unsigned i = 0; i < shape.size(); i++)
    permutedShape.push_back(shape[permuteDims[i]]);
  permutedType = inType.getWithSizesAndDtype(llvm::ArrayRef(permutedShape),
                                             inType.getOptionalDtype());
  return success();
}

Type Torch::getDefaultAccType(PatternRewriter &rewriter, Type inputType) {
  if (inputType.isF16())
    return rewriter.getF32Type();
  if (inputType.isBF16())
    return rewriter.getF32Type();
  if (isa<Float32Type>(inputType))
    return rewriter.getF32Type();
  if (isa<Float64Type>(inputType))
    return rewriter.getF64Type();
  if (isa<Float8E5M2Type>(inputType))
    return rewriter.getF32Type();
  if (isa<Float8E4M3FNType>(inputType))
    return rewriter.getF32Type();
  if (isa<Float8E5M2FNUZType>(inputType))
    return rewriter.getF32Type();
  if (isa<Float8E4M3FNUZType>(inputType))
    return rewriter.getF32Type();
  if (inputType.isInteger(8))
    // this is an intentional deviation from CUDA (which accumulates i8 to i64)
    return rewriter.getI32Type();
  if (inputType.isInteger(16))
    return rewriter.getI64Type();
  if (inputType.isInteger(32))
    return rewriter.getI64Type();
  if (inputType.isInteger(64))
    return rewriter.getI64Type();
  return inputType;
}
