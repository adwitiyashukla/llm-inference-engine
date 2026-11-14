#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace infer {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message);

template <class... Args>
std::string strf(const char* fmt, Args... args) {
  int n = std::snprintf(nullptr, 0, fmt, args...);
  if (n <= 0) return {};
  std::string out(static_cast<size_t>(n), '\0');
  std::snprintf(out.data(), out.size() + 1, fmt, args...);
  return out;
}

double now_seconds();

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
uint16_t f32_to_bf16(float f);

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

void* aligned_malloc(size_t bytes, size_t alignment);
void aligned_free(void* p);

template <class T>
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(size_t count) { resize(count); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      aligned_free(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  ~Buffer() { aligned_free(data_); }

  void resize(size_t count) {
    aligned_free(data_);
    data_ = nullptr;
    size_ = count;
    if (count == 0) return;
    data_ = static_cast<T*>(aligned_malloc(count * sizeof(T), 64));
    std::memset(static_cast<void*>(data_), 0, count * sizeof(T));
  }
  void ensure(size_t count) {
    if (count > size_) resize(count);
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

std::string read_text_file(const std::string& path);
bool path_exists(const std::string& path);
bool is_directory(const std::string& path);
std::string join_path(const std::string& a, const std::string& b);

class Rng {
 public:
  explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) { reseed(seed); }
  void reseed(uint64_t seed);
  uint64_t next();
  float uniform();
  float normal();

 private:
  uint64_t s_[4] = {};
};

}
