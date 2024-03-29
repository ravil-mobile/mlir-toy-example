cmake_minimum_required(VERSION 3.20.0)
project(toyc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIR})

set(LLVM_TARGET_DEFINITIONS mlir/ToyOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyTableGen)

set(LLVM_TARGET_DEFINITIONS mlir/ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(InterfaceIncGen)

set(SRC toyc.cpp
        parser/AST.cpp
        mlir/Dialect.cpp
        mlir/MLIRGen.cpp
        mlir/ToyCombine.cpp
        mlir/ShapeInferencePass.cpp
        mlir/LowerToAffineLoops.cpp
        mlir/LowerToLLVM.cpp
        )
add_executable(${CMAKE_PROJECT_NAME} ${SRC})

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
  LLVMObject
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  LLVMOrcJIT
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
  ${dialect_libs}
  ${extension_libs}
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRCallInterfaces
  #MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRExecutionEngine
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "-fno-rtti")
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include ${CMAKE_BINARY_DIR})

add_dependencies(${CMAKE_PROJECT_NAME} ToyTableGen InterfaceIncGen)
