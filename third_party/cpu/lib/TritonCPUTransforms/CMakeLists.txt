# set_source_files_properties(${MLIR_DIR} PROPERTIES COMPILE_FLAGS -Wnoerror=comment )
add_triton_library(TritonCPUTransforms
    ConvertDotOp/ConvertDotCommon.cpp
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    ConvertDotOp/ConvertDotToFMA.cpp
    Canonicalize.cpp
    ConvertDotProduct.cpp
    ConvertDotToOneDNN.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)

set_source_files_properties(ConvertDotToOneDNN.cpp PROPERTIES COMPILE_FLAGS "-Wall")
