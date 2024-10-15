#include <cpu/x64/amx_tile_configure.hpp>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/brgemm/brgemm_types.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>

using namespace dnnl::impl::cpu::x64;

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
// dummy definition for DNNL lite linkage
__attribute__((weak)) void print_verbose_header() {}
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

static constexpr int PALETTE_SIZE = 64;
static constexpr int DEFAULT_KERNEL_SIZE = 1024;
static constexpr int MAX_KERNEL_SIZE = 2048;

using read_lock_guard_t = std::shared_lock<std::shared_mutex>;
using write_lock_guard_t = std::unique_lock<std::shared_mutex>;
static std::shared_mutex g_brgemm_lock;

struct brgemm_cache_info_t {
  brgemm_desc_t desc;
  brgemm_kernel_t *kernel;
  std::unique_ptr<char[]> palette;
};

static std::vector<brgemm_cache_info_t> g_cache(DEFAULT_KERNEL_SIZE);
static int64_t g_kernel_id = -1;

// TODO(haixin): use syscall to determine page size?
static constexpr size_t SCRATCH_SIZE = 2 * 4096;
// TODO(haixin): need to use custom thread management for scratch in the future?
static thread_local char scratch[SCRATCH_SIZE] = {0};

extern "C" {

int64_t dnnl_brgemm_dispatch(int64_t M, int64_t N, int64_t K, int64_t LDA,
                             int64_t LDB, int64_t LDC, int64_t stride_a,
                             int64_t stride_b, float beta, int64_t dtypeA,
                             int64_t dtypeB) {
  auto dnnl_dtypeA = static_cast<dnnl_data_type_t>(dtypeA);
  auto dnnl_dtypeB = static_cast<dnnl_data_type_t>(dtypeB);
  int64_t dtypeA_size = dnnl::impl::types::data_type_size(dnnl_dtypeA);
  int64_t dtypeB_size = dnnl::impl::types::data_type_size(dnnl_dtypeB);
  brgemm_strides_t stride_info{stride_a * dtypeA_size, stride_b * dtypeB_size};

  write_lock_guard_t g(g_brgemm_lock);
  g_kernel_id++;
  assert(g_kernel_id < MAX_KERNEL_SIZE &&
         "Too many brgemm kernels are created");
  if (g_kernel_id >= DEFAULT_KERNEL_SIZE) {
    if (g_kernel_id >= (int64_t)g_cache.size()) {
      g_cache.resize(g_kernel_id + 1);
    }
  }

  dnnl::impl::status_t status = brgemm_desc_init(
      &g_cache[g_kernel_id].desc, cpu_isa_t::isa_undef,
      brgemm_batch_kind_t::brgemm_strd, dnnl_dtypeA, dnnl_dtypeB,
      /*transA=*/false, /*transB=*/false, brgemm_layout_t::brgemm_row_major,
      1.0f, beta, LDA, LDB, LDC, M, N, K, &stride_info);
  assert(status == dnnl::impl::status::success &&
         "Failed to initialize BRGEMM descriptor");

  status = brgemm_kernel_create(&g_cache[g_kernel_id].kernel,
                                g_cache[g_kernel_id].desc);
  assert(status == dnnl::impl::status::success &&
         "Failed to JIT BRGEMM kernel");

  brgemm_attr_t dnnl_attrs;
  brgemm_desc_set_attr(&g_cache[g_kernel_id].desc, dnnl_attrs);

  if (g_cache[g_kernel_id].desc.is_tmm) {
    g_cache[g_kernel_id].palette.reset(new char[PALETTE_SIZE]);
    status = brgemm_init_tiles(g_cache[g_kernel_id].desc,
                               g_cache[g_kernel_id].palette.get());
    assert(status == dnnl::impl::status::success &&
           "Failed to initialize palette for BRGEMM");
  }

  return g_kernel_id;
}

void dnnl_brgemm_tileconfig(int64_t kernel_idx) {
  std::unique_ptr<read_lock_guard_t> lock_guard;
  if (kernel_idx >= DEFAULT_KERNEL_SIZE) {
    lock_guard = std::make_unique<read_lock_guard_t>(g_brgemm_lock);
  }
  assert(kernel_idx >= 0 && kernel_idx < (int64_t)g_cache.size() &&
         "Invalid kernel handler");
  brgemm_desc_t &desc = g_cache[kernel_idx].desc;
  if (!desc.is_tmm) {
    return;
  }
  char *palette_buffer = g_cache[kernel_idx].palette.get();
  assert(palette_buffer != nullptr && "Invalid palette for BRGEMM kernel");
  amx_tile_configure(palette_buffer);
}

void dnnl_brgemm_tilerelease() {
  if (!mayiuse(avx512_core_amx)) {
    return;
  }

  amx_tile_release();
}

void dnnl_brgemm_execute(int64_t kernel_idx, void *A, uint64_t A_offset,
                         void *B, uint64_t B_offset, void *C, uint64_t C_offset,
                         int num) {
  std::unique_ptr<read_lock_guard_t> lock_guard;
  if (kernel_idx >= DEFAULT_KERNEL_SIZE) {
    lock_guard = std::make_unique<read_lock_guard_t>(g_brgemm_lock);
  }
  assert(kernel_idx >= 0 && kernel_idx < (int64_t)g_cache.size() &&
         "Invalid kernel handler");
  brgemm_desc_t &desc = g_cache[kernel_idx].desc;
  brgemm_kernel_t *kernel = g_cache[kernel_idx].kernel;
  assert(kernel && "Invalid brgemm kernel pointer");
  size_t A_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc.dt_a) * A_offset;
  size_t B_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc.dt_b) * B_offset;
  size_t C_offset_in_bytes =
      dnnl::impl::types::data_type_size(desc.dt_c) * C_offset;
  char *A_arith = static_cast<char *>(A) + A_offset_in_bytes;
  char *B_arith = static_cast<char *>(B) + B_offset_in_bytes;
  char *C_arith = static_cast<char *>(C) + C_offset_in_bytes;
  brgemm_kernel_execute(kernel, num, A_arith, B_arith, nullptr, C_arith,
                        scratch);
}
}