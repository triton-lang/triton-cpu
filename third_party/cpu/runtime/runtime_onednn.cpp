#include <cpu/x64/amx_tile_configure.hpp>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/brgemm/brgemm_types.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>

#include <oneapi/dnnl/dnnl_types.h>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <oneapi/dnnl/dnnl_ukernel_types.h>

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

std::map<> runtimeCahe;

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

void *create_brgemm_ukernel(int64_t M, int64_t N, int64_t K_k,
                            int64_t batch_size, int64_t lda, int64_t ldb,
                            int64_t ldc, int64_t dtypeA, int64_t dtypeB,
                            int64_t dtypeC) {
  using K = std::array<int64_t, 10>;
  K key{M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC};

  static std::map<K, dnnl::ukernel::brgemm> savedUkernels;
  {
    read_lock_guard_t r_g(g_brgemm_lock);
    if (savedUkernel.count(key) != 0) {
      return &savedUkernel.find(key)->second;
    }
  }

  write_lock_guard_t w_g(g_brgemm_lock);

  if (savedUkernel.count(key) != 0) {
    return &savedUkernel.find(key)->second;
  }

  auto dnnl_dtypeA = static_cast<dnnl_data_type_t>(dtypeA);
  auto dnnl_dtypeB = static_cast<dnnl_data_type_t>(dtypeB);
  auto dnnl_dtypeC = static_cast<dnnl_data_type_t>(dtypeC);

  dnnl::ukernel::brgemm brg;
  brg = brgemm(M, N, K_k, batch_size, lda, ldb, ldc, dnnl_dtypeA, dnnl_dtypeB,
               dnnl_dtypeC);

  // Instruct the kernel to append the result to C tensor.
  brg.set_add_C(true);
  // Finalize the initialization.
  brg.finalize();
  // Generate the executable JIT code for the objects.
  brg.generate();

  auto it = savedUkernels.insert({key, brg});
  return &it.first->second;
}

// Size of the packed tensor.
// blocked_B_size = ldb * K_k * memory::data_type_size(b_dt);

// B_blocked = new uint16_t[blocked_B_size * n_calls];
// B_base_ptr = B_blocked;

void *create_transform_ukernel(int64_t K, int64_t N, int64_t in_ld,
                               int64_t out_ld, int64_t inDtype,
                               int64_t outDtype) {
  using K_t = std::array<int64_t, 6>;
  K_t key{K, N, in_ld, out_ld, inDtype, outDtype};

  static std::map<K_t, dnnl::ukernel::transform> savedUkernels;
  {
    read_lock_guard_t r_g(g_brgemm_lock);
    if (savedUkernel.count(key) != 0) {
      return &savedUkernel.find(key)->second;
    }
  }

  write_lock_guard_t w_g(g_brgemm_lock);

  if (savedUkernel.count(key) != 0) {
    return &savedUkernel.find(key)->second;
  }

  // Packing B tensor routine. The BRGeMM ukernel expects B passed in a
  // special VNNI format for low precision data types, e.g., bfloat16_t.
  // Note: the routine doesn't provide a `batch_size` argument in the
  // constructor as it can be either incorporated into `K` dimension, or
  // manually iterated over in a for-loop on the user side.
  dnnl::ukernel::transform pack_B(/* K = */ K_k * n_calls, /* N = */ N,
                                  /* in_pack_type = */ pack_type::no_trans,
                                  /* in_ld = */ N,
                                  /* out_ld = */ ldb, /* in_dt = */ b_dt,
                                  /* out_dt = */ b_dt);

  // Pack B routine execution.
  // Note: usually should be split to process only that part of B that the
  // ukernel will execute.
  pack_B.generate();

  auto it = savedUkernels.insert({key, pack_B});
  return &it.first->second;
}

void call_transform(const void *transform_k, const void *original_data,
                    void *blocked_data) {
  auto &pack_B = reinterpret_cast<dnnl::ukernel::transform *>(transform_k);
  pack_B.execute(original_data, blocked_data);
}

// Most questionable function - no shure where to leave forming of offsets lists
// maybe too difficult in client code
void prepare_buffers(int64_t batch_size) {
  // BRGeMM ukernel execute section.
  // Prepare buffers for execution.
  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(batch_size);
  for (memory::dim i = 0; i < batch_size; i++) {
    const memory::dim A_offset_i = i * K_k * a_dt_size;
    const memory::dim B_offset_i =
        need_pack ? i * blocked_B_size : i * N * K_k * b_dt_size;
    A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
  }
}

// for perf targets
void prepare_hw_context(const void *brg_k) {
  auto &brg = reinterpret_cast<dnnl::ukernel::brgemm *>(brg_k);
  brg.set_hw_context();
}

void call_brgemm(const void *brg_k, const void *A_ptr, const void *B_ptr,
                 void *C_ptr) {
  auto &brg = reinterpret_cast<dnnl::ukernel::brgemm *>(brg_k);

  // An execute call. `A_B` is a vector of pointers to A and packed B
  // tensors. `acc_ptr` is a pointer to an accumulator buffer.
  brg.execute(A_ptr, B_ptr, A_B_offsets, C_ptr, scratch);
}

// at the end of execution
void release_hw_context() {
  // Once all computations are done, need to release HW context.
  dnnl::ukernel::brgemm::release_hw_context();
}

} // extern C
