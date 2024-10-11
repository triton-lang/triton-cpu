add_triton_library(TritonCPUXsmm
    ConvertVectorToXsmm.cpp
    VnniUtils.cpp
    ValueUtils.cpp
    XsmmEnum.cpp
    XsmmUtils.cpp

    DEPENDS
    TritonCPUXsmmPassIncGen
    TritonCPUXsmmAttrDefIncGen
    xsmm

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRVectorDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRInferTypeOpInterface
    MLIRLinalgUtils
    xsmm
)

target_include_directories(TritonCPUXsmm
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
)
