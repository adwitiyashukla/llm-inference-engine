#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace infer {

class ThreadPool {
 public:
  explicit ThreadPool(int threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int threads() const { return static_cast<int>(workers_.size()) + 1; }
  void set_static(bool on) { static_schedule_ = on; }
  bool is_static() const { return static_schedule_; }

  template <class F>
  void parallel_for(int64_t n, int64_t grain, F&& fn) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    if (workers_.empty() || n <= grain) {
      fn(int64_t{0}, n);
      return;
    }
    using Fn = std::remove_reference_t<F>;
    Job job{n, grain,
            [](void* ctx, int64_t begin, int64_t end) { (*static_cast<Fn*>(ctx))(begin, end); },
            const_cast<void*>(static_cast<const void*>(&fn))};
    run(job);
  }

 private:
  struct Job {
    int64_t n = 0;
    int64_t grain = 1;
    void (*call)(void*, int64_t, int64_t) = nullptr;
    void* ctx = nullptr;
  };

  void run(const Job& job);
  void work(int index);
  void loop(int index);

  std::vector<std::thread> workers_;
  Job job_;
  std::atomic<int64_t> next_{0};
  std::atomic<int> pending_{0};
  std::atomic<uint64_t> generation_{0};
  std::atomic<int> sleepers_{0};
  std::atomic<bool> stop_{false};
  bool static_schedule_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
};

int default_thread_count();

}
