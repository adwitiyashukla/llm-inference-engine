#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "check.h"
#include "common.h"
#include "kernels.h"
#include "quant.h"

using namespace infer;

namespace {

std::vector<float> random_vector(Rng& rng, int64_t n, float scale) {
  std::vector<float> v(static_cast<size_t>(n));
  for (float& x : v) x = rng.normal() * scale;
  return v;
}

void run_matmul(const KernelSet& k, const WeightView& w, const float* x, int64_t tokens, float* y,
                int64_t r0, int64_t r1) {
  const int64_t nb = w.cols / kQBlock;
  std::vector<int8_t> xq(static_cast<size_t>(tokens * w.cols));
  std::vector<float> xd(static_cast<size_t>(tokens * std::max<int64_t>(nb, 1)));
  std::vector<int32_t> xs(xd.size());
  if (wtype_quantized(w.type)) {
    for (int64_t t = 0; t < tokens; ++t) {
      k.quantize_row(x + t * w.cols, w.cols, xq.data() + t * w.cols, xd.data() + t * nb,
                     xs.data() + t * nb);
    }
  }
  MatmulTask task;
  task.w = &w;
  task.row_begin = r0;
  task.row_end = r1;
  task.tokens = tokens;
  task.x = x;
  task.x_stride = w.cols;
  task.xq = xq.data();
  task.xd = xd.data();
  task.xs = xs.data();
  task.xq_stride = w.cols;
  task.xd_stride = nb;
  task.y = y;
  task.y_stride = w.rows;
  k.matmul[static_cast<int>(w.type)](task);
}

const WType kAllTypes[] = {WType::F32, WType::F16, WType::BF16, WType::Q8, WType::Q4};

}

TEST(f16_round_trip_is_exact) {
  int mismatches = 0;
  for (uint32_t h = 0; h < 65536; ++h) {
    float f = f16_to_f32(static_cast<uint16_t>(h));
    if (std::isnan(f)) continue;
    if (f32_to_f16(f) != h) ++mismatches;
  }
  EXPECT(mismatches == 0);
  EXPECT(f16_to_f32(0x3c00) == 1.0f);
  EXPECT(f16_to_f32(0x0001) == std::ldexp(1.0f, -24));
  EXPECT(f32_to_f16(65520.0f) == 0x7c00);
  EXPECT(f32_to_f16(std::ldexp(1.0f, -25)) == 0x0000);
  EXPECT(f32_to_f16(std::ldexp(1.5f, -25)) == 0x0001);
}

TEST(f16_conversion_matches_every_kernel_set) {
  Rng rng(11);
  std::vector<float> x;
  for (int i = 0; i < 4096; ++i) x.push_back(rng.normal() * std::ldexp(1.0f, i % 40 - 20));
  for (int i = 0; i < 64; ++i) x.push_back(std::ldexp(1.0f + static_cast<float>(i) / 1024.0f, -24));
  for (int i = 0; i < 64; ++i) x.push_back(1.0f + std::ldexp(static_cast<float>(i), -11));
  x.push_back(70000.0f);
  x.push_back(-70000.0f);
  std::vector<uint16_t> want(x.size());
  for (size_t i = 0; i < x.size(); ++i) want[i] = f32_to_f16(x[i]);
  for (const KernelSet* k : supported_kernels()) {
    std::vector<uint16_t> got(x.size());
    k->to_f16(x.data(), got.data(), static_cast<int64_t>(x.size()));
    EXPECT(got == want);
  }
}

TEST(bf16_rounds_to_nearest_even) {
  EXPECT(f32_to_bf16(1.0f) == 0x3f80);
  float tie_down = bf16_to_f32(0x3f80) + std::ldexp(1.0f, -8);
  EXPECT(f32_to_bf16(tie_down) == 0x3f80);
  float tie_up = bf16_to_f32(0x3f81) + std::ldexp(1.0f, -8);
  EXPECT(f32_to_bf16(tie_up) == 0x3f82);
  for (uint32_t h = 0; h < 65536; h += 7) {
    float f = bf16_to_f32(static_cast<uint16_t>(h));
    if (!std::isnan(f)) EXPECT(f32_to_bf16(f) == h);
  }
}

TEST(activation_quantization_is_identical_across_kernel_sets) {
  Rng rng(3);
  const int64_t n = 4096;
  std::vector<float> x = random_vector(rng, n, 2.0f);
  for (int i = 0; i < 32; ++i) x[static_cast<size_t>(i)] = 0.0f;
  x[40] = 1e-30f;
  std::vector<int8_t> q0(n);
  std::vector<float> d0(n / kQBlock);
  std::vector<int32_t> s0(n / kQBlock);
  scalar_kernels()->quantize_row(x.data(), n, q0.data(), d0.data(), s0.data());
  for (int64_t b = 0; b < n / kQBlock; ++b) {
    int32_t sum = 0;
    for (int64_t i = 0; i < kQBlock; ++i) sum += q0[b * kQBlock + i];
    EXPECT(sum == s0[b]);
  }
  for (const KernelSet* k : supported_kernels()) {
    std::vector<int8_t> q(n);
    std::vector<float> d(n / kQBlock);
    std::vector<int32_t> s(n / kQBlock);
    k->quantize_row(x.data(), n, q.data(), d.data(), s.data());
    EXPECT(q == q0);
    EXPECT(s == s0);
    EXPECT(std::memcmp(d.data(), d0.data(), d.size() * sizeof(float)) == 0);
  }
}

TEST(q8_and_q4_weights_round_trip) {
  Rng rng(5);
  const int64_t rows = 16;
  const int64_t cols = 512;
  std::vector<float> w = random_vector(rng, rows * cols, 0.05f);
  for (WType type : {WType::Q8, WType::Q4}) {
    OwnedWeight ow;
    make_weight(ow, w.data(), rows, cols, type);
    double err = 0.0;
    double ref = 0.0;
    std::vector<float> row(cols);
    for (int64_t r = 0; r < rows; ++r) {
      dequantize_row(ow.view, r, row.data());
      for (int64_t c = 0; c < cols; ++c) {
        double e = row[c] - w[r * cols + c];
        err += e * e;
        ref += static_cast<double>(w[r * cols + c]) * w[r * cols + c];
      }
    }
    double rel = std::sqrt(err / ref);
    EXPECT(rel < (type == WType::Q8 ? 0.01 : 0.12));
  }
}

TEST(grouped_layout_round_trips_every_row) {
  Rng rng(7);
  const int64_t rows = 37;
  const int64_t cols = 96;
  const int64_t nb = cols / kQBlock;
  std::vector<float> w = random_vector(rng, rows * cols, 0.05f);
  for (WType type : {WType::Q8, WType::Q4}) {
    OwnedWeight ow;
    make_weight(ow, w.data(), rows, cols, type);
    EXPECT(ow.data.size() == static_cast<size_t>(48) * wtype_row_bytes(type, cols));
    std::vector<int8_t> want(cols);
    std::vector<uint16_t> want_s(nb);
    std::vector<int8_t> got(cols);
    std::vector<uint16_t> got_s(nb);
    for (int64_t r = 0; r < rows; ++r) {
      if (type == WType::Q8) {
        quantize_q8_row(w.data() + r * cols, cols, want.data(), want_s.data());
      } else {
        quantize_q4_row(w.data() + r * cols, cols, want.data(), want_s.data());
      }
      load_quant_row(ow.view, r, got.data(), got_s.data());
      EXPECT(got == want);
      EXPECT(got_s == want_s);
    }
    for (int64_t r = rows; r < 48; ++r) {
      load_quant_row(ow.view, r, got.data(), got_s.data());
      EXPECT(std::all_of(got.begin(), got.end(), [](int8_t v) { return v == 0; }));
      EXPECT(std::all_of(got_s.begin(), got_s.end(), [](uint16_t v) { return v == 0; }));
    }
  }
}

TEST(q4_scale_search_beats_plain_rounding) {
  Rng rng(9);
  const int64_t n = 32 * 256;
  std::vector<float> x = random_vector(rng, n, 1.0f);
  std::vector<int8_t> q(n);
  std::vector<uint16_t> s(n / kQBlock);
  quantize_q4_row(x.data(), n, q.data(), s.data());
  std::vector<float> back(n);
  for (int64_t i = 0; i < n; ++i) back[i] = f16_to_f32(s[i / kQBlock]) * q[i];
  double searched = 0.0;
  double plain = 0.0;
  for (int64_t b = 0; b < n / kQBlock; ++b) {
    const float* xb = x.data() + b * kQBlock;
    float mx = 0.0f;
    for (int i = 0; i < kQBlock; ++i) {
      if (std::fabs(xb[i]) > std::fabs(mx)) mx = xb[i];
    }
    float d = mx / -8.0f;
    for (int i = 0; i < kQBlock; ++i) {
      int v = std::clamp(static_cast<int>(std::nearbyint(xb[i] / d)), -8, 7);
      double e = xb[i] - d * v;
      plain += e * e;
      double e2 = xb[i] - back[b * kQBlock + i];
      searched += e2 * e2;
    }
  }
  EXPECT(searched < plain * 0.95);
}

TEST(matmul_matches_scalar_for_every_type) {
  Rng rng(21);
  const int64_t rows = 37;
  const int64_t cols = 320;
  const int64_t tokens = 9;
  std::vector<float> w = random_vector(rng, rows * cols, 0.05f);
  std::vector<float> x = random_vector(rng, tokens * cols, 1.0f);
  for (WType type : kAllTypes) {
    OwnedWeight ow;
    make_weight(ow, w.data(), rows, cols, type);
    std::vector<float> want(static_cast<size_t>(tokens * rows));
    run_matmul(*scalar_kernels(), ow.view, x.data(), tokens, want.data(), 0, rows);
    for (const KernelSet* k : supported_kernels()) {
      std::vector<float> got(static_cast<size_t>(tokens * rows));
      run_matmul(*k, ow.view, x.data(), tokens, got.data(), 0, rows);
      if (wtype_quantized(type)) {
        EXPECT(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
        continue;
      }
      double worst = 0.0;
      for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
      }
      if (worst > 2e-4) std::printf("  %s %s worst %.3g\n", k->name, wtype_name(type), worst);
      EXPECT(worst <= 2e-4);
    }
  }
}

TEST(int8_matmul_is_bit_identical_for_every_tile_size) {
  Rng rng(27);
  const int64_t rows = 53;
  const int64_t cols = 192;
  const int64_t max_tokens = 14;
  std::vector<float> w = random_vector(rng, rows * cols, 0.05f);
  std::vector<float> x = random_vector(rng, max_tokens * cols, 1.0f);
  for (WType type : {WType::Q8, WType::Q4}) {
    OwnedWeight ow;
    make_weight(ow, w.data(), rows, cols, type);
    std::vector<float> want(static_cast<size_t>(max_tokens * rows));
    run_matmul(*scalar_kernels(), ow.view, x.data(), max_tokens, want.data(), 0, rows);
    for (const KernelSet* k : supported_kernels()) {
      for (int64_t tokens = 1; tokens <= max_tokens; ++tokens) {
        std::vector<float> got(static_cast<size_t>(tokens * rows));
        run_matmul(*k, ow.view, x.data(), tokens, got.data(), 3, rows);
        run_matmul(*k, ow.view, x.data(), tokens, got.data(), 0, 3);
        EXPECT(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
      }
    }
  }
}

TEST(matmul_is_batch_and_partition_invariant) {
  Rng rng(33);
  const int64_t rows = 29;
  const int64_t cols = 256;
  const int64_t tokens = 7;
  std::vector<float> w = random_vector(rng, rows * cols, 0.05f);
  std::vector<float> x = random_vector(rng, tokens * cols, 1.0f);
  for (WType type : kAllTypes) {
    OwnedWeight ow;
    make_weight(ow, w.data(), rows, cols, type);
    for (const KernelSet* k : supported_kernels()) {
      std::vector<float> batched(static_cast<size_t>(tokens * rows));
      run_matmul(*k, ow.view, x.data(), tokens, batched.data(), 0, rows);
      std::vector<float> single(static_cast<size_t>(tokens * rows));
      for (int64_t t = 0; t < tokens; ++t) {
        std::vector<float> y(static_cast<size_t>(rows));
        run_matmul(*k, ow.view, x.data() + t * cols, 1, y.data(), 0, 11);
        run_matmul(*k, ow.view, x.data() + t * cols, 1, y.data(), 11, rows);
        std::copy(y.begin(), y.end(), single.begin() + t * rows);
      }
      EXPECT(std::memcmp(batched.data(), single.data(), batched.size() * sizeof(float)) == 0);
    }
  }
}

TEST(vector_kernels_match_scalar) {
  Rng rng(44);
  const int64_t n = 291;
  std::vector<float> a = random_vector(rng, n, 1.0f);
  std::vector<float> b = random_vector(rng, n, 1.0f);
  std::vector<uint16_t> bh(n);
  scalar_kernels()->to_f16(b.data(), bh.data(), n);
  const KernelSet& s = *scalar_kernels();
  for (const KernelSet* k : supported_kernels()) {
    EXPECT_NEAR(k->dot(a.data(), b.data(), n), s.dot(a.data(), b.data(), n), 1e-4);
    EXPECT_NEAR(k->dot_f16(a.data(), bh.data(), n), s.dot_f16(a.data(), bh.data(), n), 1e-4);
    std::vector<float> y1(a);
    std::vector<float> y2(a);
    k->axpy(y1.data(), 0.37f, b.data(), n);
    s.axpy(y2.data(), 0.37f, b.data(), n);
    for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-6);
    k->axpy_f16(y1.data(), -1.5f, bh.data(), n);
    s.axpy_f16(y2.data(), -1.5f, bh.data(), n);
    for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-5);
    std::vector<float> e1(a);
    std::vector<float> e2(a);
    float s1 = k->exp_shifted(e1.data(), n, 3.0f);
    float s2 = s.exp_shifted(e2.data(), n, 3.0f);
    EXPECT_NEAR(s1, s2, 1e-5 * s2);
    for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(e1[i], e2[i], 1e-6 * std::max(1.0f, e2[i]));
    std::vector<float> o1(n);
    std::vector<float> o2(n);
    std::vector<float> g = random_vector(rng, n, 4.0f);
    k->silu_mul(g.data(), b.data(), o1.data(), n);
    s.silu_mul(g.data(), b.data(), o2.data(), n);
    for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(o1[i], o2[i], 1e-5);
    std::vector<float> m1(a);
    std::vector<float> m2(a);
    k->sigmoid_mul(m1.data(), g.data(), n);
    s.sigmoid_mul(m2.data(), g.data(), n);
    for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(m1[i], m2[i], 1e-5);
  }
}

TEST(deltanet_step_matches_double_reference) {
  Rng rng(55);
  const int64_t dk = 128;
  const int64_t dv = 128;
  std::vector<float> state0 = random_vector(rng, dk * dv, 0.1f);
  std::vector<double> ref(state0.begin(), state0.end());
  std::vector<float> q = random_vector(rng, dk, 1.0f);
  std::vector<float> k = random_vector(rng, dk, 1.0f);
  std::vector<float> v = random_vector(rng, dv, 1.0f);
  const float decay = 0.93f;
  const float beta = 0.61f;
  std::vector<double> mem(dv, 0.0);
  for (int64_t i = 0; i < dk; ++i) {
    for (int64_t j = 0; j < dv; ++j) {
      ref[i * dv + j] *= decay;
      mem[j] += ref[i * dv + j] * k[i];
    }
  }
  std::vector<double> want(dv, 0.0);
  for (int64_t i = 0; i < dk; ++i) {
    for (int64_t j = 0; j < dv; ++j) {
      ref[i * dv + j] += k[i] * (v[j] - mem[j]) * beta;
      want[j] += ref[i * dv + j] * q[i];
    }
  }
  for (const KernelSet* ks : supported_kernels()) {
    std::vector<float> state(state0);
    std::vector<float> out(dv);
    std::vector<float> scratch(dv);
    ks->deltanet_step(state.data(), q.data(), k.data(), v.data(), decay, beta, out.data(),
                      scratch.data(), dk, dv);
    for (int64_t j = 0; j < dv; ++j) EXPECT_NEAR(out[j], want[j], 1e-3);
    double worst = 0.0;
    for (int64_t i = 0; i < dk * dv; ++i) worst = std::max(worst, std::fabs(state[i] - ref[i]));
    EXPECT(worst < 1e-4);
  }
}

TEST(deltanet_sequence_matches_token_steps) {
  Rng rng(66);
  const int64_t dk = 128;
  const int64_t dv = 128;
  const int64_t tokens = 9;
  const int64_t stride = 3 * dk + 8;
  std::vector<float> qkv = random_vector(rng, tokens * stride, 0.3f);
  std::vector<float> gates(static_cast<size_t>(tokens * 4));
  for (int64_t t = 0; t < tokens; ++t) {
    gates[t * 4] = 0.2f + 0.05f * static_cast<float>(t);
    gates[t * 4 + 1] = 0.97f - 0.01f * static_cast<float>(t);
  }
  std::vector<float> state0 = random_vector(rng, dk * dv, 0.1f);
  for (const KernelSet* ks : supported_kernels()) {
    std::vector<float> a(state0);
    std::vector<float> b(state0);
    std::vector<float> out_a(static_cast<size_t>(tokens * dv));
    std::vector<float> out_b(static_cast<size_t>(tokens * dv));
    std::vector<float> scratch(dv);
    for (int64_t t = 0; t < tokens; ++t) {
      const float* row = qkv.data() + t * stride;
      ks->deltanet_step(a.data(), row, row + dk, row + 2 * dk, gates[t * 4 + 1], gates[t * 4],
                        out_a.data() + t * dv, scratch.data(), dk, dv);
    }
    DeltaSeq seq;
    seq.tokens = tokens;
    seq.q = qkv.data();
    seq.k = qkv.data() + dk;
    seq.v = qkv.data() + 2 * dk;
    seq.qkv_stride = stride;
    seq.gates = gates.data();
    seq.gate_stride = 4;
    seq.out = out_b.data();
    seq.out_stride = dv;
    ks->deltanet_seq(b.data(), seq, scratch.data(), dk, dv);
    EXPECT(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    EXPECT(std::memcmp(out_a.data(), out_b.data(), out_a.size() * sizeof(float)) == 0);
  }
}

int main() {
  std::printf("cpu: %s [%s]\n", cpu_info().brand.c_str(), cpu_feature_string().c_str());
  for (const KernelSet* k : supported_kernels()) std::printf("kernel set: %s\n", k->name);
  return check::run_all();
}
