#include "kernels.h"

#include <atomic>
#include <cstdlib>

#include "common.h"

#if defined(INFER_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace infer {

const char* wtype_name(WType type) {
  switch (type) {
    case WType::F32:
      return "f32";
    case WType::F16:
      return "f16";
    case WType::BF16:
      return "bf16";
    case WType::Q8:
      return "q8";
    case WType::Q4:
      return "q4";
  }
  return "?";
}

bool parse_wtype(std::string_view name, WType& out) {
  static const WType all[] = {WType::F32, WType::F16, WType::BF16, WType::Q8, WType::Q4};
  for (WType t : all) {
    if (name == wtype_name(t)) {
      out = t;
      return true;
    }
  }
  return false;
}

size_t wtype_row_bytes(WType type, int64_t cols) {
  switch (type) {
    case WType::F32:
      return static_cast<size_t>(cols) * 4;
    case WType::F16:
    case WType::BF16:
      return static_cast<size_t>(cols) * 2;
    case WType::Q8:
      return static_cast<size_t>(cols);
    case WType::Q4:
      return static_cast<size_t>(cols) / 2;
  }
  return 0;
}

size_t WeightView::bytes() const {
  const int64_t n = stored_rows(type, rows);
  size_t total = static_cast<size_t>(n) * wtype_row_bytes(type, cols);
  if (wtype_quantized(type)) total += static_cast<size_t>(n * (cols / kQBlock)) * 2;
  return total;
}

#if defined(INFER_X86)
static void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub));
  for (int i = 0; i < 4; ++i) r[i] = static_cast<uint32_t>(regs[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

static uint64_t xgetbv0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t eax = 0;
  uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}
#endif

static CpuInfo detect_cpu() {
  CpuInfo info;
#if defined(INFER_X86)
  uint32_t r[4];
  cpuid(0, 0, r);
  uint32_t max_leaf = r[0];
  cpuid(1, 0, r);
  bool fma = (r[2] >> 12) & 1u;
  bool osxsave = (r[2] >> 27) & 1u;
  bool avx = (r[2] >> 28) & 1u;
  bool f16c = (r[2] >> 29) & 1u;
  bool os_avx = false;
  bool os_avx512 = false;
  if (osxsave) {
    uint64_t xcr0 = xgetbv0();
    os_avx = (xcr0 & 6u) == 6u;
    os_avx512 = os_avx && (xcr0 & 0xe0u) == 0xe0u;
  }
  bool avx2 = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool avx512_vnni = false;
  bool avx_vnni = false;
  if (max_leaf >= 7) {
    cpuid(7, 0, r);
    avx2 = (r[1] >> 5) & 1u;
    avx512f = (r[1] >> 16) & 1u;
    avx512bw = (r[1] >> 30) & 1u;
    avx512vl = (r[1] >> 31) & 1u;
    avx512_vnni = (r[2] >> 11) & 1u;
    cpuid(7, 1, r);
    avx_vnni = (r[0] >> 4) & 1u;
  }
  info.avx2 = os_avx && avx && avx2;
  info.fma = os_avx && avx && fma;
  info.f16c = os_avx && avx && f16c;
  info.avx_vnni = info.avx2 && avx_vnni;
  info.avx512f = os_avx512 && avx512f;
  info.avx512bw = os_avx512 && avx512bw;
  info.avx512vl = os_avx512 && avx512vl;
  info.avx512_vnni = os_avx512 && avx512_vnni;
  cpuid(0x80000000u, 0, r);
  if (r[0] >= 0x80000004u) {
    char brand[49] = {};
    for (uint32_t i = 0; i < 3; ++i) {
      cpuid(0x80000002u + i, 0, r);
      std::memcpy(brand + i * 16, r, 16);
    }
    info.brand = brand;
    size_t b = info.brand.find_first_not_of(' ');
    size_t e = info.brand.find_last_not_of(' ');
    info.brand = b == std::string::npos ? std::string() : info.brand.substr(b, e - b + 1);
  }
#endif
  if (info.brand.empty()) info.brand = "unknown cpu";
  return info;
}

const CpuInfo& cpu_info() {
  static const CpuInfo info = detect_cpu();
  return info;
}

std::string cpu_feature_string() {
  const CpuInfo& c = cpu_info();
  std::string out;
  auto add = [&](bool on, const char* name) {
    if (!on) return;
    if (!out.empty()) out += ' ';
    out += name;
  };
  add(c.avx2, "AVX2");
  add(c.fma, "FMA");
  add(c.f16c, "F16C");
  add(c.avx_vnni, "AVX-VNNI");
  add(c.avx512f, "AVX512F");
  add(c.avx512bw, "AVX512BW");
  add(c.avx512vl, "AVX512VL");
  add(c.avx512_vnni, "AVX512-VNNI");
  if (out.empty()) out = "none";
  return out;
}

#if !defined(INFER_X86)
const KernelSet* avx2_kernels() { return nullptr; }
const KernelSet* vnni_kernels() { return nullptr; }
const KernelSet* avx512_kernels() { return nullptr; }
#endif

std::vector<const KernelSet*> supported_kernels() {
  std::vector<const KernelSet*> out{scalar_kernels()};
  const CpuInfo& c = cpu_info();
  bool base = c.avx2 && c.fma && c.f16c;
  if (avx2_kernels() && base) out.push_back(avx2_kernels());
  if (vnni_kernels() && base && c.avx_vnni) out.push_back(vnni_kernels());
  if (avx512_kernels() && base && c.avx512f && c.avx512bw && c.avx512vl && c.avx512_vnni) {
    out.push_back(avx512_kernels());
  }
  return out;
}

static std::atomic<const KernelSet*> g_kernels{nullptr};

static const KernelSet* pick_kernels(std::string_view name) {
  std::vector<const KernelSet*> list = supported_kernels();
  if (name.empty() || name == "auto") return list.back();
  for (const KernelSet* k : list) {
    if (name == k->name) return k;
  }
  fail("kernel set '" + std::string(name) + "' is not supported on this cpu");
}

const KernelSet& kernels() {
  const KernelSet* k = g_kernels.load(std::memory_order_acquire);
  if (k == nullptr) {
    const char* env = std::getenv("INFER_KERNELS");
    k = pick_kernels(env != nullptr ? env : "auto");
    g_kernels.store(k, std::memory_order_release);
  }
  return *k;
}

void use_kernels(std::string_view name) {
  g_kernels.store(pick_kernels(name), std::memory_order_release);
}

}
