#include <algorithm>
#include <cmath>

#include <functional>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include <oneapi/dnnl/dnnl_types.h>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <oneapi/dnnl/dnnl_ukernel_types.h>

#include <map>
#include <mutex>
#include <shared_mutex>

using namespace dnnl;
using namespace dnnl::ukernel;

using tag = memory::format_tag;
using dt = memory::data_type;

using read_lock_guard_t = std::shared_lock<std::shared_mutex>;
using write_lock_guard_t = std::unique_lock<std::shared_mutex>;
static std::shared_mutex g_brgemm_lock;

#ifdef Nope
static inline int64_t getDnnlDataTypeVal(RewriterBase &rewriter,
                                         Attribute attr) {
  auto context = rewriter.getContext();
  auto tattr = dyn_cast_or_null<TypeAttr>(attr);
  assert(tattr);
  if (tattr == TypeAttr::get(FloatType::getF32(context))) {
    return static_cast<int64_t>(dnnl_f32);
  } else if (tattr == TypeAttr::get(FloatType::getF64(context))) {
    return static_cast<int64_t>(dnnl_f64);
  } else if (tattr == TypeAttr::get(FloatType::getBF16(context))) {
    return static_cast<int64_t>(dnnl_bf16);
  } else if (tattr == TypeAttr::get(FloatType::getF16(context))) {
    return static_cast<int64_t>(dnnl_f16);
  } else if (tattr == TypeAttr::get(
                          IntegerType::get(context, 32, IntegerType::Signed))) {
    return static_cast<int64_t>(dnnl_s32);
  } else if (tattr ==
             TypeAttr::get(IntegerType::get(context, 8, IntegerType::Signed))) {
    return static_cast<int64_t>(dnnl_s8);
  } else if (tattr == TypeAttr::get(IntegerType::get(context, 8,
                                                     IntegerType::Unsigned))) {
    return static_cast<int64_t>(dnnl_u8);
  }
  return static_cast<int64_t>(dnnl_data_type_undef);
}

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
#endif

inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
  return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
                         std::multiplies<dnnl::memory::dim>());
}

void *create_brgemm_ukernel(int64_t M, int64_t N, int64_t K_k,
                            int64_t batch_size, int64_t lda, int64_t ldb,
                            int64_t ldc, int64_t dtypeA, int64_t dtypeB,
                            int64_t dtypeC) {
  using K = std::array<int64_t, 10>;
  std::cout << "Args: M - " << M << ", N - " << N << ", K - " << K_k
            << ", bath - " << batch_size << ", lda - " << lda << ", ldb - "
            << ldb << ", ldc - " << ldc << ", dtype a - " << dtypeA
            << ", dtype b - " << dtypeB << ", dtype c - " << dtypeC << "\n";
  K key{M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC};

  static std::map<K, dnnl::ukernel::brgemm> savedUkernels;
  {
    read_lock_guard_t r_g(g_brgemm_lock);
    if (savedUkernels.count(key) != 0) {
      return &savedUkernels.find(key)->second;
    }
  }

  write_lock_guard_t w_g(g_brgemm_lock);

  if (savedUkernels.count(key) != 0) {
    return &savedUkernels.find(key)->second;
  }

  auto dnnl_dtypeA = static_cast<dnnl::memory::data_type>(dtypeA);
  auto dnnl_dtypeB = static_cast<dnnl::memory::data_type>(dtypeB);
  auto dnnl_dtypeC = static_cast<dnnl::memory::data_type>(dtypeC);

  std::cout << std::boolalpha;
  std::cout << "Is fp32? " << (dnnl_dtypeA == dt::f32) << "\n";

  dnnl::ukernel::brgemm brg;
  brg = dnnl::ukernel::brgemm(M, N, K_k, batch_size, lda, ldb, ldc, dnnl_dtypeA,
                              dnnl_dtypeB, dnnl_dtypeC);

  std::cout << "Brg: " << &brg << "\n";
  // Instruct the kernel to append the result to C tensor.
  brg.set_add_C(true);
  // Finalize the initialization.
  brg.finalize();
  // Generate the executable JIT code for the objects.
  brg.generate();

  auto it = savedUkernels.insert({key, brg});
  std::cout << "Ptr: " << &it.first->second << "\n";
  return &it.first->second;
}

void *create_transform_ukernel(int64_t K, int64_t N, int64_t in_ld,
                               int64_t out_ld, int64_t inDtype,
                               int64_t outDtype) {
  using K_t = std::array<int64_t, 6>;
  K_t key{K, N, in_ld, out_ld, inDtype, outDtype};

  static std::map<K_t, dnnl::ukernel::transform> savedUkernels;
  {
    read_lock_guard_t r_g(g_brgemm_lock);
    if (savedUkernels.count(key) != 0) {
      return &savedUkernels.find(key)->second;
    }
  }

  write_lock_guard_t w_g(g_brgemm_lock);

  if (savedUkernels.count(key) != 0) {
    return &savedUkernels.find(key)->second;
  }

  // Packing B tensor routine. The BRGeMM ukernel expects B passed in a
  // special VNNI format for low precision data types, e.g., bfloat16_t.
  // Note: the routine doesn't provide a `batch_size` argument in the
  // constructor as it can be either incorporated into `K` dimension, or
  // manually iterated over in a for-loop on the user side.
  dnnl::ukernel::transform pack_B(
      /* K = */ K, /* N = */ N,
      /* in_pack_type = */ pack_type::no_trans,
      /* in_ld = */ in_ld,
      /* out_ld = */ out_ld,
      /* in_dt = */ static_cast<dnnl::memory::data_type>(inDtype),
      /* out_dt = */ static_cast<dnnl::memory::data_type>(outDtype));

  // Pack B routine execution.
  // Note: usually should be split to process only that part of B that the
  // ukernel will execute.
  pack_B.generate();

  auto it = savedUkernels.insert({key, pack_B});
  return &it.first->second;
}

void call_all(const void *transform_k, const void *brg_k, void *A_ptr,
              void *original_B_ptr, void *C_ptr, void *scratchpad,
              int64_t A_step_in_bytes, int64_t B_step_in_bytes,
              int64_t B_block_size_in_bytes, int64_t num_batches,
              bool skip_packing = false) {

  void *blocked_data = original_B_ptr;
  std::cout << "Call Transform: " << transform_k << " Brg: " << brg_k
            << ", a: " << A_ptr << ", b: " << original_B_ptr << ", c: " << C_ptr
            << ", scr: " << scratchpad << "\n";
  std::cout << "steps: " << A_step_in_bytes << " " << B_step_in_bytes << " "
            << B_block_size_in_bytes << " n: " << num_batches << "\n";

  auto pack_B = reinterpret_cast<const dnnl::ukernel::transform *>(transform_k);
  auto brg = reinterpret_cast<const dnnl::ukernel::brgemm *>(brg_k);
  std::cout << " vanilla check pack: "
            << ((brg->get_B_pack_type() == pack_type::pack32) ? "true"
                                                              : "false")
            << "\n";
  bool need_packing =
      brg->get_B_pack_type() == pack_type::pack32 && !skip_packing;
  if (need_packing) {
    std::cout << "Will be packed. \n";
    // output

    // blocked_B_size = block_K * block_n * dtype; // ldb * K_k *
    // memory::data_type_size(b_dt);

    blocked_data = new uint8_t[B_block_size_in_bytes * num_batches];

    pack_B->execute(original_B_ptr, blocked_data);
  }

  brg->set_hw_context();

  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(num_batches);
  for (memory::dim i = 0; i < num_batches; i++) {
    const memory::dim A_offset_i =
        i * A_step_in_bytes; // * a_dt_size; // K_k * a_dt_size;
    const memory::dim B_offset_i =
        need_packing ? i * B_block_size_in_bytes : i * B_step_in_bytes;
    A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
  }

  size_t scratchpad_size = brg->get_scratchpad_size();
  std::vector<uint8_t> scratchpad_sm(scratchpad_size);

  //  An execute call. `A_B` is a vector of pointers to A and packed B
  //  tensors. `acc_ptr` is a pointer to an accumulator buffer.
  brg->execute(A_ptr, blocked_data, A_B_offsets, C_ptr, scratchpad_sm.data());

  dnnl::ukernel::brgemm::release_hw_context();

  if (need_packing) {
    delete blocked_data;
  };
}

int main() {
  // brgemm_example();
  // return 0;

  // dnnl_brgemm_dispatch(16, 16, 16, 0, 0, 0 );
  // dnnl_brgemm_tileconfig();

  // dnnl_brgemm_execute();

  // dnnl_brgemm_execute();

  // ukernel dimensions.
  // K is for a whole tensor, K_k is for a single ukernel.
  const memory::dim M = 32, K = 32, K_k = 16, N = 32;
  if (K % K_k != 0) {
    printf("K_k must divide K.\n");
    return 0;
  }
  const memory::dim n_calls = K / K_k;
  std::cout << "n_cals: " << n_calls << "\n";

  const memory::dim lda = K;
  const memory::dim ldb = N;
  const memory::dim ldc = N; // Leading dimension for accumulator.
  const memory::dim ldd = N; // Leading dimension for an actual output.
  const memory::dim batch_size = n_calls;

  memory::data_type a_dt = dt::f32; // dt::bf16;
  memory::data_type b_dt = dt::f32; // dt::bf16;
  memory::data_type c_dt = dt::f32; // Accumulator data type.
  memory::data_type d_dt = dt::f32; // Output data type.

  // A, B, and C tensors dimensions.
  memory::dims A_dims = {M, K};
  memory::dims B_dims = {K, N};
  memory::dims C_dims = {M, N};
  memory::dims D_dims = {M, N};

  // Allocate buffers with user data.
  std::vector<float> A_user_data(product(A_dims));
  std::vector<float> B_user_data(product(B_dims));
  std::vector<float> D_data(product(D_dims));      // For reference comparison
  std::vector<float> D_user_data(product(D_dims)); // For reference comparison

  // Initialize A.
  std::generate(A_user_data.begin(), A_user_data.end(), []() {
    static int i = 1;
    return i++ % 4;
  });
  // Initialize B.
  std::generate(B_user_data.begin(), B_user_data.end(), []() {
    static int i = 6;
    static int sign_gen = 0;
    int sign = (sign_gen++ % 2) ? -1 : 1;
    float val = sign * (i++ % 5);
    return val;
  });

  // Create execution dnnl::engine. Needed for reorders to operate over input
  // data.
  dnnl::engine engine(engine::kind::cpu, 0);

  // Create dnnl::stream. Needed for reorders for the same reason.
  dnnl::stream engine_stream(engine);

  // Create f32 memories. They are used as data holders and reorder into
  // memories passed to the ukernel.
  auto A_f32_md = memory::desc(A_dims, dt::f32, tag::ab);
  auto B_f32_md = memory::desc(B_dims, dt::f32, tag::ab);
  auto D_f32_md = memory::desc(D_dims, dt::f32, tag::ab);

  auto A_f32_mem = memory(A_f32_md, engine, A_user_data.data());
  auto B_f32_mem = memory(B_f32_md, engine, B_user_data.data());
  auto D_f32_mem = memory(D_f32_md, engine, D_data.data());

  // Create ukernel memories in requested data types.
  // Note that all formats are `ab`.
  auto A_md = memory::desc(A_dims, a_dt, tag::ab);
  auto B_md = memory::desc(B_dims, b_dt, tag::ab);

  auto C_md = memory::desc(C_dims, c_dt, tag::ab);
  auto D_md = memory::desc(D_dims, d_dt, tag::ab);

  auto A_mem = memory(A_md, engine);
  auto B_mem = memory(B_md, engine);

  auto C_mem = memory(C_md, engine);
  auto D_mem = memory(D_md, engine);

  const auto *A_ptr = reinterpret_cast<float *>(A_mem.get_data_handle());
  auto *B_ptr = reinterpret_cast<float *>(B_mem.get_data_handle());

  const size_t a_dt_size =
      memory::data_type_size(A_mem.get_desc().get_data_type());
  const size_t b_dt_size =
      memory::data_type_size(B_mem.get_desc().get_data_type());

  reorder(A_f32_mem, A_mem).execute(engine_stream, A_f32_mem, A_mem);
  reorder(B_f32_mem, B_mem).execute(engine_stream, B_f32_mem, B_mem);
  reorder(D_f32_mem, D_mem).execute(engine_stream, D_f32_mem, D_mem);

  float *C_ptr = reinterpret_cast<float *>(C_mem.get_data_handle());
  for (memory::dim i = 0; i < M * N; i++) {
    C_ptr[i] = 0;
  }

  auto brg_k =
      create_brgemm_ukernel(M, N, K_k, batch_size, lda, ldb, ldc, 3, 3, 3);
  auto tfrm = create_transform_ukernel(K_k * n_calls, N, N, ldb, 3, 3);

  // void *B_base_ptr = B_ptr;

  // blocked_B_size = ldb * K_k * memory::data_type_size(b_dt);

  call_all(tfrm, brg_k, (void *)A_ptr, (void *)B_ptr, (void *)C_ptr, nullptr,
           K_k * a_dt_size, N * K_k * b_dt_size,
           ldb * K_k * memory::data_type_size(b_dt), batch_size);

  printf("( m, n)  val after \"brg\" call\n");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      printf("Acc buffer(C_ptr) res: (%2d, %2d) Ref:%12f\n", m, n,
             C_ptr[m * N + n]);
      // if (scratchpad.size() != 0) {
      //   printf("Out buffer res: (%2d, %2d) Ref:%12f\n", m, n,
      //          scratchpad[m * N + n]);
      // }
    }
  }

  bool to_throw = false;
  printf("( m,  n,  k)  val\n");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      D_user_data[m * N + n] = 0;
      for (int k = 0; k < K; k++) {
        printf("(%2d, %2d, %2d) A: %12f     B: %12f\n", m, n, k,
               A_user_data[m * K + k], B_user_data[k * N + n]);
        D_user_data[m * N + n] +=
            A_user_data[m * K + k] * B_user_data[k * N + n];
      }
      const float diff = fabsf(D_user_data[m * N + n] - C_ptr[m * N + n]);
      if (diff > 1.19e-7) {
        to_throw = true;
        if (true) {
          printf("Error: (%2d, %2d) Ref:%12g Got:%12g Diff:%12g\n", m, n,
                 D_user_data[m * N + n], C_ptr[m * N + n], diff);
        }
      }
      // else {
      //   printf("Matched res: (%2d, %2d) Ref:%12f\n", m, n,
      //          D_user_data[m * N + n]);
      // }
    }
  }
  if (to_throw) {
    throw status::runtime_error;
  }

  return 0;

  // Create BRGeMM ukernel objects.
  // There are two objects:
  // * `brg` is the main one which operates over partitioned K dimension. It
  //   utilizes `beta = 1.f` to accumulate into the same buffer. It also uses
  //   `batch_size` to process as much as `n_calls - 1` iterations.
  // * `brg_po` is the ukernel that would be called the last in the chain
  //   since it has attributes attached to the object and those will execute
  //   after all accumulation over K dimension is done.
  // Note: `beta = 1.f` makes a ukernel reusable over K but will require
  // zeroing the correspondent piece of accumulation buffer.
  brgemm brg;
  if (batch_size > 0) {
    try {
      // Construct a basic brgemm object.
      brg = brgemm(M, N, K_k, batch_size, lda, ldb, ldc, a_dt, b_dt, c_dt);
      // Instruct the kernel to append the result to C tensor.
      brg.set_add_C(true);
      // Finalize the initialization.
      brg.finalize();
      // Generate the executable JIT code for the objects.
      brg.generate();
    } catch (error &e) {
      // on any other error just re-throw
      throw;
    }
  }

  // Query a scratchpad size and initialize a scratchpad buffer if the ukernel
  // is expecting it. This is a service space needed, has nothing in common
  // with accumulation buffer.
  size_t scratchpad_size = brg.get_scratchpad_size();
  std::vector<float> scratchpad(scratchpad_size);

  uint16_t *B_blocked = nullptr;
  size_t blocked_B_size = 0;

  void *B_base_ptr = B_ptr;

  // Query the packing requirement from the kernel. It's enough to query
  // packing requirements from a single object as long as only dimension
  // settings change between objects.
  // Note: example uses the one that always present regardless of dimensions.
  const bool need_pack = brg.get_B_pack_type() == pack_type::pack32;

  // If packing is needed, create a dedicated object for data transformation.
  if (need_pack) {
    // Packing B tensor routine. The BRGeMM ukernel expects B passed in a
    // special VNNI format for low precision data types, e.g., bfloat16_t.
    // Note: the routine doesn't provide a `batch_size` argument in the
    // constructor as it can be either incorporated into `K` dimension, or
    // manually iterated over in a for-loop on the user side.
    transform pack_B(/* K = */ K_k * n_calls, /* N = */ N,
                     /* in_pack_type = */ pack_type::no_trans, /* in_ld = */ N,
                     /* out_ld = */ ldb, /* in_dt = */ b_dt,
                     /* out_dt = */ b_dt);

    // Size of the packed tensor.
    blocked_B_size = ldb * K_k * memory::data_type_size(b_dt);

    B_blocked = new uint16_t[blocked_B_size * n_calls];
    B_base_ptr = B_blocked;

    // Pack B routine execution.
    // Note: usually should be split to process only that part of B that the
    // ukernel will execute.
    pack_B.generate();

    pack_B.execute(B_ptr, B_blocked);
  }

  // BRGeMM ukernel execute section.
  // Prepare buffers for execution.
  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(batch_size);
  for (memory::dim i = 0; i < batch_size; i++) {
    const memory::dim A_offset_i = i * K_k * a_dt_size;
    const memory::dim B_offset_i =
        need_pack ? i * blocked_B_size : i * N * K_k * b_dt_size;
    A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
  }

  if (brg) {
    std::cout << "brg with bs: " << batch_size << " should be called.\n";
    // Make an object to call HW specialized routines. For example, prepare
    // AMX unit.
    brg.set_hw_context();

    // An execute call. `A_B` is a vector of pointers to A and packed B
    // tensors. `acc_ptr` is a pointer to an accumulator buffer.
    brg.execute(A_ptr, B_base_ptr, A_B_offsets, C_ptr, scratchpad.data());

    printf("( m, n)  val after \"brg\" call\n");
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        printf("Acc buffer(C_ptr) res: (%2d, %2d) Ref:%12f\n", m, n,
               C_ptr[m * N + n]);
        if (scratchpad.size() != 0) {
          printf("Out buffer res: (%2d, %2d) Ref:%12f\n", m, n,
                 scratchpad[m * N + n]);
        }
      }
    }
  }

  // Once all computations are done, need to release HW context.
  brgemm::release_hw_context();

  // Clean up an extra buffer.
  delete B_blocked;

  // Used for verification results, need unconditional reorder.
  auto user_D_mem = memory(D_f32_md, engine, D_data.data());
  reorder(C_mem, user_D_mem).execute(engine_stream, C_mem, user_D_mem);

  // A simplified fast verification that ukernel returned expected results.
  // Note: potential off-by-1 or 2 errors may pop up. This could be solved
  // with more sparse filling.
  // bool to_throw = false;
  printf("( m,  n,  k)  val\n");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      D_user_data[m * N + n] = 0;
      for (int k = 0; k < K; k++) {
        printf("(%2d, %2d, %2d) A: %12f     B: %12f\n", m, n, k,
               A_user_data[m * K + k], B_user_data[k * N + n]);
        D_user_data[m * N + n] +=
            A_user_data[m * K + k] * B_user_data[k * N + n];
      }
      const float diff = fabsf(D_user_data[m * N + n] - D_data[m * N + n]);
      if (diff > 1.19e-7) {
        to_throw = true;
        if (true) {
          printf("Error: (%2d, %2d) Ref:%12g Got:%12g Diff:%12g\n", m, n,
                 D_user_data[m * N + n], D_data[m * N + n], diff);
        }
      } else {
        printf("Matched res: (%2d, %2d) Ref:%12f\n", m, n,
               D_user_data[m * N + n]);
      }
    }
  }
  if (to_throw) {
    throw status::runtime_error;
  }

  return 0;
}
