#include "common.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace infer {

void fail(const std::string& message) { throw Error(message); }

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x3ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = -1;
      do {
        mantissa <<= 1;
        ++shift;
      } while ((mantissa & 0x400u) == 0);
      mantissa &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t mantissa = x & 0x7fffffu;
  int exponent = static_cast<int>((x >> 23) & 0xffu);
  if (exponent == 0xff) {
    return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u | (mantissa >> 13) : 0u));
  }
  int e = exponent - 127 + 15;
  if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(sign);
    uint32_t m = mantissa | 0x800000u;
    int shift = 14 - e;
    uint32_t half = m >> shift;
    uint32_t rest = m & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t h = sign | (static_cast<uint32_t>(e) << 10) | (mantissa >> 13);
  uint32_t rest = mantissa & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
  return static_cast<uint16_t>(h);
}

uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  if ((x & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((x >> 16) | 0x40u);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}

void* aligned_malloc(size_t bytes, size_t alignment) {
  if (bytes == 0) bytes = alignment;
#if defined(_WIN32)
  void* p = _aligned_malloc(bytes, alignment);
#else
  void* p = nullptr;
  if (posix_memalign(&p, alignment, bytes) != 0) p = nullptr;
#endif
  if (p == nullptr) fail(strf("out of memory allocating %zu bytes", bytes));
  return p;
}

void aligned_free(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

static std::filesystem::path to_path(const std::string& utf8) {
  return std::filesystem::path(
      std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(to_path(path), std::ios::binary);
  if (!in) fail("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool path_exists(const std::string& path) {
  std::error_code ec;
  return std::filesystem::exists(to_path(path), ec);
}

bool is_directory(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_directory(to_path(path), ec);
}

std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  char last = a.back();
  if (last == '/' || last == '\\') return a + b;
  return a + "/" + b;
}

static uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

void Rng::reseed(uint64_t seed) {
  uint64_t x = seed;
  for (auto& s : s_) s = splitmix64(x);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

uint64_t Rng::next() {
  uint64_t result = rotl(s_[1] * 5, 7) * 9;
  uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

float Rng::uniform() { return static_cast<float>(next() >> 40) * (1.0f / 16777216.0f); }

float Rng::normal() {
  float u1 = uniform();
  float u2 = uniform();
  if (u1 < 1e-7f) u1 = 1e-7f;
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

}
