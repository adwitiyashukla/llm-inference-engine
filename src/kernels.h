#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace infer {

enum class WType : uint8_t { F32 = 0, F16 = 1, BF16 = 2, Q8 = 3, Q4 = 4 };

constexpr int64_t kQBlock = 32;
constexpr int64_t kRowGroup = 16;

const char* wtype_name(WType type);
bool parse_wtype(std::string_view name, WType& out);
size_t wtype_row_bytes(WType type, int64_t cols);
inline bool wtype_quantized(WType type) { return type == WType::Q8 || type == WType::Q4; }
inline int64_t stored_rows(WType type, int64_t rows) {
  return wtype_quantized(type) ? (rows + kRowGroup - 1) / kRowGroup * kRowGroup : rows;
}

struct WeightView {
  WType type = WType::F32;
  int64_t rows = 0;
  int64_t cols = 0;
  const void* data = nullptr;
  const uint16_t* scales = nullptr;
  bool valid() const { return data != nullptr; }
  size_t bytes() const;
};

struct MatmulTask {
  const WeightView* w = nullptr;
  int64_t row_begin = 0;
  int64_t row_end = 0;
  int64_t tokens = 0;
  const float* x = nullptr;
  int64_t x_stride = 0;
  const int8_t* xq = nullptr;
  const float* xd = nullptr;
  const int32_t* xs = nullptr;
  int64_t xq_stride = 0;
  int64_t xd_stride = 0;
  float* y = nullptr;
  int64_t y_stride = 0;
};

using MatmulFn = void (*)(const MatmulTask&);

struct DeltaSeq {
  int64_t tokens = 0;
  const float* q = nullptr;
  const float* k = nullptr;
  const float* v = nullptr;
  int64_t qkv_stride = 0;
  const float* gates = nullptr;
  int64_t gate_stride = 0;
  float* out = nullptr;
  int64_t out_stride = 0;
};

struct KernelSet {
  const char* name;
  MatmulFn matmul[5];
  void (*quantize_row)(const float* x, int64_t n, int8_t* q, float* d, int32_t* s);
  float (*dot)(const float* a, const float* b, int64_t n);
  float (*dot_f16)(const float* a, const uint16_t* b, int64_t n);
  void (*axpy)(float* y, float a, const float* x, int64_t n);
  void (*axpy_f16)(float* y, float a, const uint16_t* x, int64_t n);
  void (*to_f16)(const float* x, uint16_t* y, int64_t n);
  float (*exp_shifted)(float* x, int64_t n, float shift);
  void (*silu)(float* x, int64_t n);
  void (*silu_mul)(const float* gate, const float* up, float* out, int64_t n);
  void (*sigmoid_mul)(float* x, const float* gate, int64_t n);
  void (*deltanet_step)(float* state, const float* q, const float* k, const float* v, float decay,
                        float beta, float* out, float* scratch, int64_t dk, int64_t dv);
  void (*deltanet_seq)(float* state, const DeltaSeq& seq, float* scratch, int64_t dk, int64_t dv);
};

struct CpuInfo {
  std::string brand;
  bool avx2 = false;
  bool fma = false;
  bool f16c = false;
  bool avx_vnni = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool avx512_vnni = false;
};

const CpuInfo& cpu_info();
std::string cpu_feature_string();

const KernelSet* scalar_kernels();
const KernelSet* avx2_kernels();
const KernelSet* vnni_kernels();
const KernelSet* avx512_kernels();

const KernelSet& kernels();
std::vector<const KernelSet*> supported_kernels();
void use_kernels(std::string_view name);

}
