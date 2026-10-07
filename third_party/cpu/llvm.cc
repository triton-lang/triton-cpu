#include "triton/Tools/Sys/GetEnv.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
#include <sys/sysctl.h>
#endif

namespace py = nanobind;

namespace {

void initializeHostTarget() {
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    if (llvm::InitializeNativeTarget())
      throw std::runtime_error("LLVM native target is not available");
    if (llvm::InitializeNativeTargetAsmPrinter())
      throw std::runtime_error("LLVM native target assembly printer is not "
                               "available");
  });

  // LLVM's global thread pool is not fork-safe. Triton kernels are small, so
  // disabling LLVM's internal parallelism also avoids unnecessary overhead.
  llvm::parallel::strategy = llvm::hardware_concurrency(1);
}

void setLLVMBooleanOption(const std::string &name, bool value) {
  auto options = llvm::cl::getRegisteredOptions();
  auto it = options.find(name);
  if (it == options.end())
    return;
  it->second->addOccurrence(1, name, value ? "true" : "false");
}

// Return true if the CPU triple is AArch64.  Used by the NEON fallback below.
static bool isAArch64() {
  std::string triple = llvm::sys::getProcessTriple();
  std::size_t separator = triple.find('-');
  if (separator == std::string::npos)
    return false;
  std::string arch = triple.substr(0, separator);
  return arch == "aarch64" || arch == "arm64";
}

namespace arm {

bool osSupportsBF16() {
#if defined(__linux__) && defined(__aarch64__)
  return (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  int value = 0;
  size_t size = sizeof(value);
  return sysctlbyname("hw.optional.arm.FEAT_BF16", &value, &size, nullptr, 0) ==
             0 &&
         size == sizeof(value) && value == 1;
#else
  return false;
#endif
}

} // namespace arm

// Share Arm feature fallbacks between the Python backend and LLVM target
// machines. LLVM may omit BF16 even when the operating system reports it.
std::set<std::string> getCPUFeatures() {
  auto features = llvm::sys::getHostCPUFeatures();

  std::set<std::string> result;
  for (const auto &feature : features)
    if (feature.second)
      result.insert(feature.first().str());

  if (isAArch64()) {
    // Apply this before BF16 so an empty LLVM result still gets mandatory NEON.
    if (result.empty())
      result.insert("neon");
    // Only query the OS when LLVM has not already reported BF16.
    if (!result.count("bf16") && arm::osSupportsBF16())
      result.insert("bf16");
  }

  return result;
}

// Build a comma-separated feature string (e.g. "+sve,+sve2,+neon") suitable
// for passing to TargetMachine::createTargetMachine, derived from
// getCPUFeatures() so both share the same NEON fallback logic.
static std::string getCPUFeaturesString() {
  std::string result;
  for (const auto &feature : getCPUFeatures()) {
    if (!result.empty())
      result += ",";
    result += "+";
    result += feature;
  }
  return result;
}

std::unique_ptr<llvm::TargetMachine>
createHostTargetMachine(llvm::Module &module, const std::string &cpuFeatures,
                        bool enableFpFusion, bool enableFastMath) {
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  llvm::TargetOptions options;
  if (enableFpFusion)
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  if (enableFastMath)
    options.NoTrappingFPMath = true;
  options.TrapUnreachable = true;
  options.MCOptions.AsmVerbose = true;
  options.MCOptions.PreserveAsmComments = true;

  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  return std::unique_ptr<llvm::TargetMachine>{target->createTargetMachine(
      module.getTargetTriple(), llvm::sys::getHostCPUName(), cpuFeatures,
      options, llvm::Reloc::PIC_, std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
}

// Set the vscale_range function attribute on all functions in the module when
// SVE is available.  This tells LLVM that the function may use scalable
// vectors and that the runtime vscale value is within the given range.
// For SVE, the vector length is vscale * 128 bits, and on common AArch64
// implementations vscale is 1 (128-bit) or 2 (256-bit).  We use a conservative
// range of [1, 16] to cover all SVE implementations (up to 2048-bit).
void setVscaleRangeForSVE(llvm::Module &module, llvm::StringRef cpuFeatures) {
  if (cpuFeatures.find("+sve") == std::string::npos)
    return;

  for (llvm::Function &function : module.functions()) {
    if (function.isIntrinsic())
      continue;
    function.addFnAttr(llvm::Attribute::getWithVScaleRangeArgs(
        function.getContext(), /*Min=*/1, /*Max=*/16));
  }
}

std::string translateHostLLVMIRToASM(llvm::Module &module, bool enableFpFusion,
                                     bool enableFastMath) {
  if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP"))
    setLLVMBooleanOption("print-after-all", true);

  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<llvm::StringRef, 3> flags;
      llvm::StringRef(flagList).split(flags, ',');
      for (llvm::StringRef flag : flags)
        setLLVMBooleanOption(flag.str(), true);
    }
  }

  for (llvm::Function &function : module.functions())
    if (!function.hasFnAttribute(llvm::Attribute::NoInline))
      function.addFnAttr(llvm::Attribute::AlwaysInline);

  llvm::legacy::PassManager modulePasses;
  modulePasses.add(llvm::createAlwaysInlinerLegacyPass());
  modulePasses.add(llvm::createVerifierPass());

  const bool enableTiming =
      mlir::triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enableTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  modulePasses.run(module);

  llvm::SmallString<0> timePassesStr;
  llvm::raw_svector_ostream reportStream(timePassesStr);
  if (enableTiming) {
    llvm::reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }

  module.setTargetTriple(llvm::Triple(llvm::sys::getProcessTriple()));
  // Propagate host CPU features to the target machine and set vscale_range
  // attribute on functions when SVE is available so that scalable vector
  // code generation is enabled.
  const std::string cpuFeatures = getCPUFeaturesString();
  setVscaleRangeForSVE(module, cpuFeatures);
  auto machine = createHostTargetMachine(module, cpuFeatures, enableFpFusion,
                                         enableFastMath);
  module.setDataLayout(machine->createDataLayout());

  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream bufferedStream(stream);
    llvm::legacy::PassManager codegenPasses;
    machine->addPassesToEmitFile(codegenPasses, bufferedStream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile);
    codegenPasses.run(module);

    if (enableTiming) {
      llvm::reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }
  return result;
}

void setHostTarget(llvm::Module &module) {
  initializeHostTarget();
  module.setTargetTriple(llvm::Triple(llvm::sys::getProcessTriple()));

  // Propagate host CPU features to the target machine and set vscale_range
  // attribute on functions when SVE is available so that scalable vector
  // code generation is enabled.
  const std::string cpuFeatures = getCPUFeaturesString();
  setVscaleRangeForSVE(module, cpuFeatures);

  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module.getTargetTriple(), llvm::sys::getHostCPUName(), cpuFeatures, {},
      llvm::Reloc::PIC_)};
  module.setDataLayout(machine->createDataLayout());
}

} // namespace

void init_triton_cpu_llvm(py::module_ &m) {
  m.def("get_cpu_triple", &llvm::sys::getProcessTriple);
  m.def("get_cpu_name", []() { return llvm::sys::getHostCPUName().str(); });
  m.def("get_cpu_features", &getCPUFeatures);
  m.def("set_host_target",
        [](llvm::Module *module) { setHostTarget(*module); });
  m.def("translate_to_asm",
        [](std::string llvmIR, bool enableFpFusion,
           bool enableFastMath) -> py::object {
          std::string result;
          {
            py::gil_scoped_release release;
            initializeHostTarget();

            llvm::LLVMContext context;
            std::unique_ptr<llvm::MemoryBuffer> buffer =
                llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
            llvm::SMDiagnostic error;
            std::unique_ptr<llvm::Module> module =
                llvm::parseIR(buffer->getMemBufferRef(), error, context);
            if (!module) {
              llvm::report_fatal_error(
                  "failed to parse IR: " + error.getMessage() +
                  "lineno: " + std::to_string(error.getLineNo()));
            }
            result = translateHostLLVMIRToASM(*module, enableFpFusion,
                                              enableFastMath);
          }
          return py::str(result.c_str(), result.size());
        });
}
