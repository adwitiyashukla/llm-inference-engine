#include "thread_pool.h"

#include <algorithm>

#if defined(INFER_X86)
#include <immintrin.h>
#endif

namespace infer {

namespace {

constexpr int kSpinLimit = 1 << 16;
constexpr int kYieldAfter = 1 << 10;

inline void cpu_relax() {
#if defined(INFER_X86)
  _mm_pause();
#elif defined(__aarch64__) && !defined(_MSC_VER)
  __asm__ __volatile__("yield");
#else
  std::this_thread::yield();
#endif
}

}

ThreadPool::ThreadPool(int threads) {
  if (threads < 1) threads = 1;
  workers_.reserve(static_cast<size_t>(threads - 1));
  for (int i = 1; i < threads; ++i) workers_.emplace_back([this, i] { loop(i); });
}

ThreadPool::~ThreadPool() {
  stop_.store(true, std::memory_order_seq_cst);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    generation_.fetch_add(1, std::memory_order_seq_cst);
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::run(const Job& job) {
  job_ = job;
  next_.store(0, std::memory_order_relaxed);
  pending_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
  generation_.fetch_add(1, std::memory_order_seq_cst);
  if (sleepers_.load(std::memory_order_seq_cst) > 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    cv_.notify_all();
  }
  work(0);
  for (int spins = 0; pending_.load(std::memory_order_acquire) != 0; ++spins) {
    if (spins < kYieldAfter) {
      cpu_relax();
    } else {
      std::this_thread::yield();
    }
  }
}

void ThreadPool::work(int index) {
  const Job& job = job_;
  if (static_schedule_) {
    const int64_t count = threads();
    const int64_t begin = job.n * index / count;
    const int64_t end = job.n * (index + 1) / count;
    for (int64_t b = begin; b < end; b += job.grain) {
      job.call(job.ctx, b, std::min(end, b + job.grain));
    }
    return;
  }
  for (;;) {
    int64_t b = next_.fetch_add(job.grain, std::memory_order_relaxed);
    if (b >= job.n) break;
    job.call(job.ctx, b, std::min(job.n, b + job.grain));
  }
}

void ThreadPool::loop(int index) {
  uint64_t seen = 0;
  for (;;) {
    uint64_t gen = generation_.load(std::memory_order_acquire);
    int spins = 0;
    while (gen == seen) {
      if (stop_.load(std::memory_order_relaxed)) return;
      if (++spins < kSpinLimit) {
        if (spins % kYieldAfter == 0) {
          std::this_thread::yield();
        } else {
          cpu_relax();
        }
      } else {
        std::unique_lock<std::mutex> lock(mutex_);
        sleepers_.fetch_add(1, std::memory_order_seq_cst);
        cv_.wait(lock, [&] {
          return generation_.load(std::memory_order_seq_cst) != seen ||
                 stop_.load(std::memory_order_seq_cst);
        });
        sleepers_.fetch_sub(1, std::memory_order_seq_cst);
        spins = 0;
      }
      gen = generation_.load(std::memory_order_acquire);
    }
    seen = gen;
    if (stop_.load(std::memory_order_acquire)) return;
    work(index);
    pending_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

int default_thread_count() {
  unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 4 : static_cast<int>(n);
}

}
