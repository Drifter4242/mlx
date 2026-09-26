// Copyright © 2024 Apple Inc.
#include "mlx/fence.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

// ---- fence trace ring buffer (exo fork, 2026-09-06 wedge hunt) ----------
// Every Fence::update / Fence::wait records (fence buffer ptr, op, stream,
// count, thread) here; a CPU-side spin longer than MLX_FENCE_SPIN_DUMP_S
// (default 5) dumps the last entries to stderr once, so a wedge documents
// the exact update/wait order without a debugger.
namespace {
struct FenceEv { uint64_t t_us; void* fence; uint32_t count; uint32_t stream; uint8_t op; uint8_t dev; uint64_t tid; };
constexpr size_t kRing = 4096;
FenceEv g_ring[kRing];
std::atomic<uint64_t> g_seq{0};
inline uint64_t now_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline uint64_t tid_now() { uint64_t t = 0; pthread_threadid_np(nullptr, &t); return t; }
inline void trace_ev(void* fence, uint8_t op, mlx::core::Stream s, uint32_t count) {
  auto i = g_seq.fetch_add(1, std::memory_order_relaxed) % kRing;
  g_ring[i] = FenceEv{now_us(), fence, count, (uint32_t)s.index, op, (uint8_t)(s.device == mlx::core::Device::cpu ? 0 : 1), tid_now()};
}
inline bool self_release_enabled() {
  static const bool on = getenv("MLX_FENCE_SELF_RELEASE") && atoi(getenv("MLX_FENCE_SELF_RELEASE")) != 0;
  return on;
}
void dump_trace(void* fence, uint32_t count, uint32_t value) {
  // One full dump per process (several when self-release is on, so every release is documented).
  static std::atomic<int> dumps{0};
  if (dumps.fetch_add(1) >= (self_release_enabled() ? 8 : 1)) return;
  uint64_t n = g_seq.load(); size_t k = n < kRing ? n : kRing;
  fprintf(stderr, "[mlx fence-trace] CPU spin on fence %p awaiting %u (value %u); active_tasks=%d; last %zu fence events (op 0=update 1=wait_cpu 2=wait_gpu; dev 0=cpu 1=gpu):\n",
          fence, count, value, mlx::core::scheduler::n_active_tasks(), k);
  for (uint64_t j = n - k; j < n; j++) { auto& e = g_ring[j % kRing];
    fprintf(stderr, "[mlx fence-trace] %llu t=%llu fence=%p op=%u dev=%u stream=%u count=%u tid=%llu%s\n", (unsigned long long)j, (unsigned long long)e.t_us, e.fence, e.op, e.dev, e.stream, e.count, (unsigned long long)e.tid, e.fence == fence ? "  <== THIS" : ""); }
  fflush(stderr);
}

// EXPERIMENTS ONLY (MLX_FENCE_SELF_RELEASE=1, task49): after a long CPU spin, find the ROOT of the wedge in the
// trace ring -- the earliest wait (since the last >200 ms gap, i.e. the current eval) whose awaited value is
// above the fence's current value -- and write that value, like scripts/fence_diagnose.py --apply does from
// lldb. Lets a reproducer finish and exit cleanly instead of leaving an orphaned fence_wait kernel on the GPU
// (which only a reboot clears). Output computed after a release may be wrong.
void self_release() {
  uint64_t n = g_seq.load();
  size_t k = n < kRing ? n : kRing;
  uint64_t start = n - k;
  for (uint64_t j = n - k + 1; j < n; j++) {
    if (g_ring[j % kRing].t_us - g_ring[(j - 1) % kRing].t_us > 200000) {
      start = j;
    }
  }
  for (uint64_t j = start; j < n; j++) {
    auto& e = g_ring[j % kRing];
    if (e.op == 0) {
      continue;
    }
    auto* v = static_cast<std::atomic_uint*>(static_cast<MTL::Buffer*>(e.fence)->contents());
    uint32_t cur = v[0].load();
    if (cur < e.count) {
      v[0].store(e.count);
#if defined(__aarch64__)
      __builtin_arm_dsb(0xf);
#endif
      fprintf(stderr, "[mlx fence-trace] SELF-RELEASE seq=%llu fence=%p op=%u dev=%u stream=%u value %u -> %u\n",
              (unsigned long long)j, e.fence, e.op, e.dev, e.stream, cur, e.count);
      fflush(stderr);
      return;
    }
  }
  fprintf(stderr, "[mlx fence-trace] SELF-RELEASE: no unsatisfied wait in the current eval (not the fence flavour)\n");
  fflush(stderr);
}
} // namespace

namespace mlx::core {

struct FenceImpl {
  FenceImpl(Stream stream) {
    auto d = metal::device(stream.device).mtl_device();
    if (!d->supportsFamily(MTL::GPUFamilyMetal3)) {
      use_fast = false;
    } else if (__builtin_available(macOS 15, iOS 18, *)) {
      use_fast = env::metal_fast_synch();
    }

    if (!use_fast) {
      event = std::make_unique<Event>(stream);
    } else {
      auto buf = allocator::malloc(sizeof(uint32_t)).ptr();
      fence = static_cast<void*>(buf);
      cpu_value()[0] = 0;
    }
  }

  ~FenceImpl() {
    if (use_fast) {
      allocator::free(allocator::Buffer{static_cast<MTL::Buffer*>(fence)});
    }
  }
  bool use_fast{false};
  uint32_t count{0};
  void* fence;
  std::unique_ptr<Event> event;

  std::atomic_uint* cpu_value() {
    return static_cast<std::atomic_uint*>(
        static_cast<MTL::Buffer*>(fence)->contents());
  }
};

Fence::Fence(Stream stream) {
  fence_ = std::make_shared<FenceImpl>(stream);
}

void Fence::wait(Stream stream, const array& x, uint32_t value) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());

  if (!f.use_fast) {
    auto& event = *f.event;
    event.set_value(value);
    event.wait(stream);
    return;
  }

  if (stream.device == Device::cpu) {
    trace_ev(f.fence, 1, stream, value);
    scheduler::enqueue(stream, [fence_ = fence_, value]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      uint64_t spins = 0; uint64_t t0 = now_us();
      static const uint64_t dump_after_us = (uint64_t)((getenv("MLX_FENCE_SPIN_DUMP_S") ? atof(getenv("MLX_FENCE_SPIN_DUMP_S")) : 5.0) * 1000000.0);
      while (f.cpu_value()[0] < value) {
        if ((++spins & 0xFFFF) == 0 && now_us() - t0 > dump_after_us) {
          dump_trace(f.fence, value, f.cpu_value()[0]);
          if (self_release_enabled()) {
            self_release();
            t0 = now_us();
          } else {
            t0 = UINT64_MAX / 2;
          }
        }
#if defined(__aarch64__)
        // mlx#3142: LDAR/DMB ISH is inner-shareable (CPU-only); a GPU or
        // RDMA-DMA write to the fence page can stay invisible forever and
        // this loop spins on a stale cache line. DSB SY (full system) makes
        // the device write visible. Upstream wontfix; permanent fork carry.
        __builtin_arm_dsb(0xf);
#endif
      }
    });
    return;
  }

  trace_ev(f.fence, 2, stream, value);
  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  // Register outputs to ensure that no kernels which depends on the
  // output starts before this one is done
  compute_encoder.register_output_array(x);

  auto kernel = d.get_kernel("fence_wait");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto buf = static_cast<MTL::Buffer*>(f.fence);
  compute_encoder.set_buffer(buf, 0);
  compute_encoder.set_bytes(value, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);

  compute_encoder.get_command_buffer()->addCompletedHandler(
      [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
}

uint32_t Fence::update(Stream stream, const array& x, bool cross_device) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  f.count++;
  if (f.use_fast) trace_ev(f.fence, 0, stream, f.count);

  if (!f.use_fast) {
    f.event->set_value(f.count);
    f.event->signal(stream);
    return f.count;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, count = f.count]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      f.cpu_value()[0] = count;
#if defined(__aarch64__)
      // mlx#3142 write side: seq_cst compiles to DMB ISH; the GPU-side
      // fence_wait kernel may never observe the store. DSB SY publishes it.
      __builtin_arm_dsb(0xf);
#endif
    });
    return f.count;
  }

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  // Launch input visibility kernels
  if (cross_device) {
    auto kernel = d.get_kernel("input_coherent");
    uint32_t nthreads = (x.data_size() * x.itemsize() + sizeof(uint32_t) - 1) /
        sizeof(uint32_t);
    MTL::Size group_dims = MTL::Size(1024, 1, 1);
    MTL::Size grid_dims = MTL::Size((nthreads + 1024 - 1) / 1024, 1, 1);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_bytes(nthreads, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Barrier on previous kernels
  compute_encoder.barrier();

  // Launch value update kernel
  auto kernel = d.get_kernel("fence_update");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto buf = static_cast<MTL::Buffer*>(f.fence);
  compute_encoder.set_buffer(buf, 0);
  compute_encoder.set_bytes(f.count, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);

  compute_encoder.get_command_buffer()->addCompletedHandler(
      [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
  return f.count;
}

} // namespace mlx::core
