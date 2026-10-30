add_triton_library(TritonCPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonCPUTableGen
  TritonCPUAttrDefsIncGen
  TritonGPUCGAAttrIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUOpsEnumsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
