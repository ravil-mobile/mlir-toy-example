#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<toy::TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext* ctx) : mlir::OpRewritePattern<toy::TransposeOp>(ctx, 1) {}
  mlir::LogicalResult matchAndRewrite(toy::TransposeOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto transposeInputOp = input.getDefiningOp<toy::TransposeOp>();

    if (!transposeInputOp) {
      return mlir::failure();
    }

    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return mlir::success();
  }
};

struct SimplifyReshapeReshapeOptPattern : public mlir::OpRewritePattern<toy::ReshapeOp> {
  SimplifyReshapeReshapeOptPattern(mlir::MLIRContext* ctx) : mlir::OpRewritePattern<toy::ReshapeOp>(ctx, 1) {}
  mlir::LogicalResult matchAndRewrite(toy::ReshapeOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto reshapeInputOp = input.getDefiningOp<toy::ReshapeOp>();

    if (!reshapeInputOp) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, {reshapeInputOp.getOperand()});
    return mlir::success();
  }
};

struct SimplifyRedundantReshape : public mlir::OpRewritePattern<toy::ReshapeOp> {
  SimplifyRedundantReshape(mlir::MLIRContext* ctx) : mlir::OpRewritePattern<toy::ReshapeOp>(ctx, 1) {}
  mlir::LogicalResult matchAndRewrite(toy::ReshapeOp op, mlir::PatternRewriter &rewriter) const override {
    auto reshapeInput = op.getOperand();
    auto reshapeOutput = op.getResult();

    if (reshapeOutput.getType() == reshapeInput.getType()) {
      rewriter.replaceOp(op, {reshapeInput});
      return mlir::success();
    }
    else {
      return mlir::failure();
    }
  }
};


struct FoldConstantReshapeOptPattern : public mlir::OpRewritePattern<toy::ReshapeOp> {
  FoldConstantReshapeOptPattern(mlir::MLIRContext* ctx) : mlir::OpRewritePattern<toy::ReshapeOp>(ctx, 1) {}

  mlir::LogicalResult matchAndRewrite(toy::ReshapeOp op, mlir::PatternRewriter &rewriter) const override {
    auto reshapeInput = op.getOperand();
    auto constantInputOp = reshapeInput.getDefiningOp<toy::ConstantOp>();

    if (!constantInputOp) {
      return mlir::failure();
    }

    auto attr = constantInputOp->getAttrOfType<mlir::DenseElementsAttr>("value");
    if (!attr) {
      return mlir::failure();
    }

    assert(op->getResults().size() == 1);
    mlir::Type resultType = op->getResult(0).getType();

    assert(resultType.cast<mlir::ShapedType>() != nullptr);
    auto reshapedValues = attr.reshape(resultType.cast<mlir::ShapedType>());

    auto loc = constantInputOp->getLoc();
    auto reshapedConstantOp = rewriter.create<toy::ConstantOp>(loc, resultType, reshapedValues);

    rewriter.replaceOp(op, reshapedConstantOp);  
    return mlir::success();
  };
};
}

void toy::TransposeOp::getCanonicalizationPatterns(
  mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}

void toy::ReshapeOp::getCanonicalizationPatterns(
  mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<SimplifyReshapeReshapeOptPattern, SimplifyRedundantReshape,
                FoldConstantReshapeOptPattern>(context);
}

