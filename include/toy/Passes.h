#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace toy {
std::unique_ptr<mlir::Pass> createShapeInferencePass();
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace toy

#endif // TOY_PASSES_H