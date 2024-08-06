add_triton_library(TritonCPUTransforms
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp
    ConvertDotProduct.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)
