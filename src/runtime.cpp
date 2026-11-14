#include "runtime.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace infer {

namespace {

constexpr int64_t kChunk = 256;

float* thread_buffer(size_t n) {
  thread_local std::vector<float> buf;
  if (buf.size() < n) buf.resize(n);
  return buf.data();
}

float* thread_buffer2(size_t n) {
  thread_local std::vector<float> buf;
  if (buf.size() < n) buf.resize(n);
  return buf.data();
}

void rms_norm(const float* x, float* out, const float* w, int64_t n, float eps) {
  double ss = 0.0;
  for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  float var = static_cast<float>(ss / static_cast<double>(n));
  float r = 1.0f / std::sqrt(var + eps);
  for (int64_t i = 0; i < n; ++i) out[i] = (x[i] * r) * w[i];
}

void l2_normalize(float* x, int64_t n, float scale) {
  double ss = 0.0;
  for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  float inv = 1.0f / std::sqrt(static_cast<float>(ss) + 1e-6f);
  for (int64_t i = 0; i < n; ++i) x[i] = (x[i] * inv) * scale;
}

float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

struct ChunkPartial {
  float m;
  float l;
};

}

Runtime::Timer::Timer(Runtime& rt, Phase phase)
    : rt_(rt),
      phase_(phase),
      active_(rt.profiling_ && !rt.timing_),
      start_(active_ ? now_seconds() : 0.0) {
  if (active_) rt_.timing_ = true;
}

Runtime::Timer::~Timer() {
  if (!active_) return;
  rt_.profile_[static_cast<size_t>(phase_)] += now_seconds() - start_;
  rt_.timing_ = false;
}

std::vector<std::pair<std::string, double>> Runtime::profile() const {
  static const char* const names[] = {"weight projections", "lm head",        "attention",
                                      "deltanet",           "norms and rope", "activations",
                                      "mtp draft head",     "other"};
  std::vector<std::pair<std::string, double>> out;
  for (size_t i = 0; i < profile_.size(); ++i) out.emplace_back(names[i], profile_[i]);
  return out;
}

void KvPool::init(int64_t layers, int64_t kv_heads, int64_t head_dim, int64_t block_size,
                  int64_t blocks, bool f16) {
  layers_ = layers;
  kv_heads_ = kv_heads;
  head_dim_ = head_dim;
  block_size_ = block_size;
  blocks_ = blocks;
  f16_ = f16;
  elem_ = f16 ? 2 : 4;
  size_t per_layer = static_cast<size_t>(blocks * kv_heads * block_size * head_dim) * elem_;
  k_.clear();
  v_.clear();
  k_.resize(static_cast<size_t>(layers));
  v_.resize(static_cast<size_t>(layers));
  for (int64_t l = 0; l < layers; ++l) {
    k_[static_cast<size_t>(l)].resize(per_layer);
    v_[static_cast<size_t>(l)].resize(per_layer);
  }
  free_.clear();
  for (int64_t b = blocks - 1; b >= 0; --b) free_.push_back(static_cast<int32_t>(b));
}

int32_t KvPool::allocate() {
  if (free_.empty()) return -1;
  int32_t b = free_.back();
  free_.pop_back();
  return b;
}

void KvPool::release(int32_t block) { free_.push_back(block); }

size_t KvPool::bytes() const {
  return 2 * static_cast<size_t>(layers_ * blocks_ * kv_heads_ * block_size_ * head_dim_) * elem_;
}

void Batch::add_run(Sequence* seq, const int32_t* toks, int64_t count, int64_t first_pos,
                    int64_t snapshots, bool output_all) {
  BatchRun run;
  run.seq = seq;
  run.first_row = rows();
  run.rows = count;
  run.snapshots = snapshots;
  for (int64_t i = 0; i < count; ++i) {
    tokens.push_back(toks[i]);
    positions.push_back(first_pos + i);
    if (output_all || i + 1 == count) out_rows.push_back(run.first_row + i);
  }
  runs.push_back(run);
}

Runtime::Runtime(const Model& model, ThreadPool& pool, const RuntimeOptions& options)
    : model_(model), cfg_(model.config()), pool_(pool), opt_(options) {
  const ModelConfig& c = cfg_;
  if (opt_.max_batch_tokens < 1) opt_.max_batch_tokens = 1;
  if (opt_.max_sequences < 1) opt_.max_sequences = 1;
  const int64_t blocks = (opt_.kv_tokens + opt_.block_size - 1) / opt_.block_size;
  main_kv_.init(c.attn_layers(), c.kv_heads, c.head_dim, opt_.block_size, blocks, opt_.kv_f16);
  if (model.weights().mtp.present) {
    mtp_kv_.init(1, c.kv_heads, c.head_dim, opt_.block_size, blocks, opt_.kv_f16);
  }
  sequences_.resize(static_cast<size_t>(opt_.max_sequences));
  slot_used_.assign(static_cast<size_t>(opt_.max_sequences), 0);
  const int64_t lin = c.linear_layers();
  if (lin > 0) {
    conv_state_size_ = static_cast<size_t>((c.conv_kernel - 1) * c.conv_dim());
    ssm_head_size_ = static_cast<size_t>(c.lin_k_dim * c.lin_v_dim);
    ssm_layer_size_ = static_cast<size_t>(c.lin_v_heads) * ssm_head_size_;
    ssm_block_cols_ = c.lin_v_dim % 64 == 0 ? 64 : c.lin_v_dim;
    banks_ = std::max<int64_t>(opt_.max_snapshots, 0) + 1;
    const size_t n = static_cast<size_t>(opt_.max_sequences * banks_ * lin);
    conv_banks_.resize(n * conv_state_size_);
    ssm_banks_.resize(n * ssm_layer_size_);
    live_bank_.assign(static_cast<size_t>(opt_.max_sequences), 0);
  }
  for (int64_t i = 0; i < c.rotary_dim / 2; ++i) {
    float e = static_cast<float>(2 * i) / static_cast<float>(c.rotary_dim);
    inv_freq_.push_back(1.0f / std::pow(c.rope_theta, e));
  }
  const int64_t T = opt_.max_batch_tokens;
  const int64_t H = c.hidden;
  const int64_t I = c.intermediate;
  const int64_t value_dim = c.lin_v_heads * c.lin_v_dim;
  proj_stride_ = std::max<int64_t>(c.q_rows() + 2 * c.kv_heads * c.head_dim, 2 * I);
  if (lin > 0) proj_stride_ = std::max(proj_stride_, c.conv_dim() + value_dim + 2 * c.lin_v_heads);
  const int64_t attn_width = std::max(c.heads * c.head_dim, value_dim);
  max_cols_ = std::max({H, I, c.heads * c.head_dim, value_dim, 2 * H});
  x_.resize(static_cast<size_t>(T * H));
  h_.resize(static_cast<size_t>(T * H));
  tmp_.resize(static_cast<size_t>(T * H));
  proj_.resize(static_cast<size_t>(T * proj_stride_));
  attn_.resize(static_cast<size_t>(T * attn_width));
  if (lin > 0) {
    conv_.resize(static_cast<size_t>(T * c.conv_dim()));
    gates_.resize(static_cast<size_t>(T * c.lin_v_heads * 2));
  }
  act_.resize(static_cast<size_t>(T * I));
  if (model.weights().mtp.present) mtp_in_.resize(static_cast<size_t>(T * 2 * H));
  xq_.resize(static_cast<size_t>(T * max_cols_));
  xd_.resize(static_cast<size_t>(T * (max_cols_ / kQBlock)));
  xs_.resize(static_cast<size_t>(T * (max_cols_ / kQBlock)));
  rope_cos_.resize(static_cast<size_t>(T * (c.rotary_dim / 2)));
  rope_sin_.resize(static_cast<size_t>(T * (c.rotary_dim / 2)));
  partial_items_ = 4 * static_cast<int64_t>(pool_.threads());
  max_chunks_ = (blocks * opt_.block_size + kChunk - 1) / kChunk + 1;
  partial_ml_.resize(static_cast<size_t>(partial_items_ * max_chunks_ * 2));
  partial_acc_.resize(static_cast<size_t>(partial_items_ * max_chunks_ * c.head_dim));
}

size_t Runtime::state_bytes() const {
  size_t total = main_kv_.bytes() + mtp_kv_.bytes();
  total += (conv_banks_.size() + ssm_banks_.size()) * 4;
  return total;
}

Sequence* Runtime::acquire() {
  for (size_t i = 0; i < slot_used_.size(); ++i) {
    if (slot_used_[i]) continue;
    slot_used_[i] = 1;
    Sequence& s = sequences_[i];
    s = Sequence();
    s.slot = static_cast<int32_t>(i);
    const int64_t lin = cfg_.linear_layers();
    if (lin > 0) {
      live_bank_[i] = 0;
      std::memset(conv_state(s.slot, 0), 0, static_cast<size_t>(lin) * conv_state_size_ * 4);
      std::memset(ssm_state(s.slot, 0, 0), 0, static_cast<size_t>(lin) * ssm_layer_size_ * 4);
    }
    return &s;
  }
  return nullptr;
}

void Runtime::release(Sequence* seq) {
  if (seq == nullptr || seq->slot < 0) return;
  for (int32_t b : seq->blocks) main_kv_.release(b);
  for (int32_t b : seq->mtp_blocks) mtp_kv_.release(b);
  slot_used_[static_cast<size_t>(seq->slot)] = 0;
  *seq = Sequence();
}

bool Runtime::reserve(Sequence& seq, int64_t length, bool mtp) {
  KvPool& kv = mtp ? mtp_kv_ : main_kv_;
  std::vector<int32_t>& blocks = mtp ? seq.mtp_blocks : seq.blocks;
  const int64_t need = (length + kv.block_size() - 1) / kv.block_size();
  while (static_cast<int64_t>(blocks.size()) < need) {
    int32_t b = kv.allocate();
    if (b < 0) return false;
    blocks.push_back(b);
  }
  return true;
}

float* Runtime::conv_bank(int32_t slot, int64_t bank, int64_t layer) {
  const size_t n = (static_cast<size_t>(slot * banks_ + bank)) * cfg_.linear_layers() + layer;
  return conv_banks_.data() + n * conv_state_size_;
}

float* Runtime::ssm_bank(int32_t slot, int64_t bank, int64_t layer, int64_t head) {
  const size_t n = (static_cast<size_t>(slot * banks_ + bank)) * cfg_.linear_layers() + layer;
  return ssm_banks_.data() + n * ssm_layer_size_ + static_cast<size_t>(head) * ssm_head_size_;
}

int64_t Runtime::snapshot_bank(int32_t slot, int64_t index) const {
  return (live_bank_[static_cast<size_t>(slot)] + 1 + index) % banks_;
}

float* Runtime::conv_state(int32_t slot, int64_t layer) {
  return conv_bank(slot, live_bank_[static_cast<size_t>(slot)], layer);
}

float* Runtime::ssm_state(int32_t slot, int64_t layer, int64_t head) {
  return ssm_bank(slot, live_bank_[static_cast<size_t>(slot)], layer, head);
}

float* Runtime::conv_snapshot(int32_t slot, int64_t index, int64_t layer) {
  return conv_bank(slot, snapshot_bank(slot, index), layer);
}

float* Runtime::ssm_snapshot(int32_t slot, int64_t index, int64_t layer, int64_t head) {
  return ssm_bank(slot, snapshot_bank(slot, index), layer, head);
}

void Runtime::restore_snapshot(Sequence& seq, int64_t index) {
  if (cfg_.linear_layers() == 0) return;
  if (index < 0 || index >= opt_.max_snapshots) fail("snapshot index out of range");
  live_bank_[static_cast<size_t>(seq.slot)] = snapshot_bank(seq.slot, index);
}

void Runtime::matmul(const float* x, int64_t x_stride, int64_t tokens, const Projection* outs,
                     size_t count) {
  Timer timer(*this, Phase::Projections);
  const KernelSet& K = kernels();
  const int64_t cols = outs[0].w->cols;
  bool quantized = false;
  int64_t total = 0;
  int64_t offsets[8];
  if (count > 8) fail("too many fused projections");
  for (size_t m = 0; m < count; ++m) {
    if (outs[m].w->cols != cols) fail("fused projections must share their input size");
    quantized = quantized || wtype_quantized(outs[m].w->type);
    offsets[m] = total;
    total += (outs[m].w->rows + kRowGroup - 1) / kRowGroup;
  }
  const int64_t xq_stride = max_cols_;
  const int64_t xd_stride = max_cols_ / kQBlock;
  if (quantized) {
    int64_t grain = std::max<int64_t>(1, tokens / (4 * pool_.threads()));
    pool_.parallel_for(tokens, grain, [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) {
        K.quantize_row(x + t * x_stride, cols, xq_.data() + t * xq_stride,
                       xd_.data() + t * xd_stride, xs_.data() + t * xd_stride);
      }
    });
  }
  const int64_t threads = pool_.threads();
  const int64_t grain = std::max<int64_t>(1, (total + threads * 8 - 1) / (threads * 8));
  pool_.parallel_for(total, grain, [&](int64_t u0, int64_t u1) {
    for (size_t m = 0; m < count; ++m) {
      const int64_t rows = outs[m].w->rows;
      const int64_t lo = offsets[m];
      const int64_t hi = lo + (rows + kRowGroup - 1) / kRowGroup;
      const int64_t a = std::max(u0, lo);
      const int64_t b = std::min(u1, hi);
      if (a >= b) continue;
      MatmulTask task;
      task.w = outs[m].w;
      task.row_begin = (a - lo) * kRowGroup;
      task.row_end = std::min((b - lo) * kRowGroup, rows);
      task.tokens = tokens;
      task.x = x;
      task.x_stride = x_stride;
      task.xq = xq_.data();
      task.xd = xd_.data();
      task.xs = xs_.data();
      task.xq_stride = xq_stride;
      task.xd_stride = xd_stride;
      task.y = outs[m].y;
      task.y_stride = outs[m].stride;
      K.matmul[static_cast<int>(outs[m].w->type)](task);
    }
  });
}

void Runtime::prepare(const Batch& batch) {
  Timer timer(*this, Phase::Other);
  const int64_t T = batch.rows();
  if (T == 0) fail("empty batch");
  if (T > opt_.max_batch_tokens) fail("batch exceeds max_batch_tokens");
  row_run_.assign(static_cast<size_t>(T), -1);
  for (size_t r = 0; r < batch.runs.size(); ++r) {
    const BatchRun& run = batch.runs[r];
    for (int64_t i = 0; i < run.rows; ++i)
      row_run_[static_cast<size_t>(run.first_row + i)] = static_cast<int64_t>(r);
    if (run.snapshots > opt_.max_snapshots || run.snapshots >= run.rows + 1)
      fail("too many snapshots");
  }
  for (int64_t t = 0; t < T; ++t) {
    if (row_run_[static_cast<size_t>(t)] < 0) fail("batch row without a sequence");
  }
  const int64_t half = cfg_.rotary_dim / 2;
  for (int64_t t = 0; t < T; ++t) {
    float pos = static_cast<float>(batch.positions[static_cast<size_t>(t)]);
    for (int64_t i = 0; i < half; ++i) {
      float f = pos * inv_freq_[static_cast<size_t>(i)];
      rope_cos_[static_cast<size_t>(t * half + i)] = std::cos(f);
      rope_sin_[static_cast<size_t>(t * half + i)] = std::sin(f);
    }
  }
}

void Runtime::rope(float* x, int64_t row) const {
  const int64_t half = cfg_.rotary_dim / 2;
  const float* c = rope_cos_.data() + row * half;
  const float* s = rope_sin_.data() + row * half;
  for (int64_t i = 0; i < half; ++i) {
    float x0 = x[i];
    float x1 = x[i + half];
    x[i] = x0 * c[i] - x1 * s[i];
    x[i + half] = x1 * c[i] + x0 * s[i];
  }
}

void Runtime::embed_rows(const Batch& batch) {
  Timer timer(*this, Phase::Other);
  const int64_t H = cfg_.hidden;
  const WeightView& e = model_.weights().embed;
  pool_.parallel_for(batch.rows(), 8, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      int32_t tok = batch.tokens[static_cast<size_t>(t)];
      if (tok < 0 || tok >= cfg_.vocab) fail("token id out of range");
      dequantize_row(e, tok, x_.data() + t * H);
    }
  });
}

void Runtime::norm_rows(const float* in, float* out, const std::vector<float>& w, int64_t rows) {
  Timer timer(*this, Phase::Norms);
  const int64_t H = cfg_.hidden;
  const float eps = cfg_.rms_eps;
  pool_.parallel_for(rows, 4, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) rms_norm(in + t * H, out + t * H, w.data(), H, eps);
  });
}

void Runtime::add_rows(float* dst, const float* src, int64_t rows) {
  Timer timer(*this, Phase::Other);
  const int64_t n = rows * cfg_.hidden;
  for (int64_t i = 0; i < n; ++i) dst[i] += src[i];
}

void Runtime::attend(const Batch& batch, KvPool& kv, int64_t layer, bool mtp, float* q_base,
                     int64_t q_stride, int64_t q_head_stride) {
  const KernelSet& K = kernels();
  const int64_t T = batch.rows();
  const int64_t nh = cfg_.heads;
  const int64_t hd = cfg_.head_dim;
  const int64_t group = cfg_.heads / cfg_.kv_heads;
  const int64_t bs = kv.block_size();
  const bool f16 = kv.f16();
  const size_t es = f16 ? 2 : 4;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)));
  float* out_base = attn_.data();
  const int64_t out_stride = nh * hd;

  auto partial = [&](int64_t t, int64_t h, int64_t c, float* acc) -> ChunkPartial {
    const BatchRun& run = batch.runs[static_cast<size_t>(row_run_[static_cast<size_t>(t)])];
    const std::vector<int32_t>& blocks = mtp ? run.seq->mtp_blocks : run.seq->blocks;
    const int64_t pos = batch.positions[static_cast<size_t>(t)];
    const int64_t k0 = c * kChunk;
    const int64_t k1 = std::min(k0 + kChunk, pos + 1);
    const int64_t g = h / group;
    const float* q = q_base + t * q_stride + h * q_head_stride;
    float* scores = thread_buffer(kChunk);
    for (int64_t j = k0; j < k1; ++j) {
      const uint8_t* kp =
          kv.key(layer, blocks[static_cast<size_t>(j / bs)], g) + (j % bs) * hd * es;
      float d = f16 ? K.dot_f16(q, reinterpret_cast<const uint16_t*>(kp), hd)
                    : K.dot(q, reinterpret_cast<const float*>(kp), hd);
      scores[j - k0] = d * scale;
    }
    float m = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < k1 - k0; ++j) m = std::max(m, scores[j]);
    float l = K.exp_shifted(scores, k1 - k0, m);
    for (int64_t i = 0; i < hd; ++i) acc[i] = 0.0f;
    for (int64_t j = k0; j < k1; ++j) {
      const uint8_t* vp =
          kv.value(layer, blocks[static_cast<size_t>(j / bs)], g) + (j % bs) * hd * es;
      if (f16) {
        K.axpy_f16(acc, scores[j - k0], reinterpret_cast<const uint16_t*>(vp), hd);
      } else {
        K.axpy(acc, scores[j - k0], reinterpret_cast<const float*>(vp), hd);
      }
    }
    return {m, l};
  };
  auto merge = [hd](float& M, float& L, float* O, ChunkPartial p, const float* acc) {
    float m = std::max(M, p.m);
    float a = std::exp(M - m);
    float b = std::exp(p.m - m);
    L = L * a + p.l * b;
    for (int64_t i = 0; i < hd; ++i) O[i] = O[i] * a + acc[i] * b;
    M = m;
  };
  auto chunks_of = [&](int64_t t) { return batch.positions[static_cast<size_t>(t)] / kChunk + 1; };

  if (T * nh <= partial_items_) {
    std::vector<int64_t> start(static_cast<size_t>(T * nh + 1), 0);
    for (int64_t i = 0; i < T * nh; ++i)
      start[static_cast<size_t>(i + 1)] = start[static_cast<size_t>(i)] + chunks_of(i / nh);
    const int64_t work = start.back();
    if (chunks_of(T - 1) > max_chunks_) fail("context exceeds the kv cache");
    pool_.parallel_for(work, 1, [&](int64_t w0, int64_t w1) {
      for (int64_t w = w0; w < w1; ++w) {
        int64_t item =
            static_cast<int64_t>(std::upper_bound(start.begin(), start.end(), w) - start.begin()) -
            1;
        int64_t c = w - start[static_cast<size_t>(item)];
        int64_t t = item / nh;
        int64_t h = item % nh;
        size_t slot = static_cast<size_t>(item * max_chunks_ + c);
        ChunkPartial p = partial(t, h, c, partial_acc_.data() + slot * hd);
        partial_ml_[2 * slot] = p.m;
        partial_ml_[2 * slot + 1] = p.l;
      }
    });
    pool_.parallel_for(T * nh, 1, [&](int64_t i0, int64_t i1) {
      for (int64_t item = i0; item < i1; ++item) {
        int64_t t = item / nh;
        int64_t h = item % nh;
        float* O = out_base + t * out_stride + h * hd;
        size_t base = static_cast<size_t>(item * max_chunks_);
        float M = partial_ml_[2 * base];
        float L = partial_ml_[2 * base + 1];
        std::memcpy(O, partial_acc_.data() + base * hd, static_cast<size_t>(hd) * 4);
        for (int64_t c = 1; c < chunks_of(t); ++c) {
          size_t slot = base + static_cast<size_t>(c);
          merge(M, L, O, {partial_ml_[2 * slot], partial_ml_[2 * slot + 1]},
                partial_acc_.data() + slot * hd);
        }
        for (int64_t i = 0; i < hd; ++i) O[i] = O[i] / L;
      }
    });
    return;
  }
  pool_.parallel_for(T * nh, 1, [&](int64_t i0, int64_t i1) {
    float* acc = thread_buffer2(static_cast<size_t>(hd));
    for (int64_t item = i0; item < i1; ++item) {
      int64_t t = item / nh;
      int64_t h = item % nh;
      float* O = out_base + t * out_stride + h * hd;
      ChunkPartial first = partial(t, h, 0, O);
      float M = first.m;
      float L = first.l;
      for (int64_t c = 1; c < chunks_of(t); ++c) {
        ChunkPartial p = partial(t, h, c, acc);
        merge(M, L, O, p, acc);
      }
      for (int64_t i = 0; i < hd; ++i) O[i] = O[i] / L;
    }
  });
}

void Runtime::attention_block(const LayerWeights& L, const Batch& batch, bool mtp) {
  const KernelSet& K = kernels();
  const ModelConfig& c = cfg_;
  const int64_t T = batch.rows();
  const int64_t H = c.hidden;
  const int64_t nh = c.heads;
  const int64_t nkv = c.kv_heads;
  const int64_t hd = c.head_dim;
  const int64_t qr = c.q_rows();
  const int64_t kr = nkv * hd;
  const int64_t P = proj_stride_;
  float* proj = proj_.data();
  Projection outs[3] = {
      {&L.attn.q, proj, P}, {&L.attn.k, proj + qr, P}, {&L.attn.v, proj + qr + kr, P}};
  matmul(h_.data(), H, T, outs, 3);
  KvPool& kv = mtp ? mtp_kv_ : main_kv_;
  const int64_t layer = mtp ? 0 : L.attn_index;
  const int64_t qhs = c.attn_gate ? 2 * hd : hd;
  const int64_t bs = kv.block_size();
  const bool f16 = kv.f16();
  const size_t es = f16 ? 2 : 4;
  const float eps = c.rms_eps;
  std::optional<Timer> phase;
  phase.emplace(*this, Phase::Norms);
  pool_.parallel_for(T * (nh + nkv), 8, [&](int64_t i0, int64_t i1) {
    for (int64_t item = i0; item < i1; ++item) {
      int64_t t = item / (nh + nkv);
      int64_t j = item % (nh + nkv);
      float* row = proj + t * P;
      if (j < nh) {
        float* q = row + j * qhs;
        rms_norm(q, q, L.attn.q_norm.data(), hd, eps);
        rope(q, t);
        continue;
      }
      int64_t g = j - nh;
      float* k = row + qr + g * hd;
      float* v = row + qr + kr + g * hd;
      rms_norm(k, k, L.attn.k_norm.data(), hd, eps);
      rope(k, t);
      const BatchRun& run = batch.runs[static_cast<size_t>(row_run_[static_cast<size_t>(t)])];
      const std::vector<int32_t>& blocks = mtp ? run.seq->mtp_blocks : run.seq->blocks;
      int64_t pos = batch.positions[static_cast<size_t>(t)];
      int32_t block = blocks[static_cast<size_t>(pos / bs)];
      uint8_t* kd = kv.key(layer, block, g) + (pos % bs) * hd * es;
      uint8_t* vd = kv.value(layer, block, g) + (pos % bs) * hd * es;
      if (f16) {
        K.to_f16(k, reinterpret_cast<uint16_t*>(kd), hd);
        K.to_f16(v, reinterpret_cast<uint16_t*>(vd), hd);
      } else {
        std::memcpy(kd, k, static_cast<size_t>(hd) * 4);
        std::memcpy(vd, v, static_cast<size_t>(hd) * 4);
      }
    }
  });
  phase.emplace(*this, Phase::Attention);
  attend(batch, kv, layer, mtp, proj, P, qhs);
  if (c.attn_gate) {
    pool_.parallel_for(T * nh, 16, [&](int64_t i0, int64_t i1) {
      for (int64_t item = i0; item < i1; ++item) {
        int64_t t = item / nh;
        int64_t h = item % nh;
        K.sigmoid_mul(attn_.data() + t * nh * hd + h * hd, proj + t * P + h * 2 * hd + hd, hd);
      }
    });
  }
  phase.reset();
  Projection o = {&L.attn.o, tmp_.data(), H};
  matmul(attn_.data(), nh * hd, T, &o, 1);
  add_rows(x_.data(), tmp_.data(), T);
}

void Runtime::linear_block(const LayerWeights& L, const Batch& batch) {
  const KernelSet& K = kernels();
  const ModelConfig& c = cfg_;
  const int64_t T = batch.rows();
  const int64_t H = c.hidden;
  const int64_t kh = c.lin_k_heads;
  const int64_t vh = c.lin_v_heads;
  const int64_t kd = c.lin_k_dim;
  const int64_t vd = c.lin_v_dim;
  const int64_t key_dim = kh * kd;
  const int64_t value_dim = vh * vd;
  const int64_t conv_dim = c.conv_dim();
  const int64_t ck = c.conv_kernel;
  const int64_t P = proj_stride_;
  const int64_t li = L.linear_index;
  float* proj = proj_.data();
  float* conv = conv_.data();
  Projection outs[4] = {{&L.lin.qkv, proj, P},
                        {&L.lin.z, proj + conv_dim, P},
                        {&L.lin.b, proj + conv_dim + value_dim, P},
                        {&L.lin.a, proj + conv_dim + value_dim + vh, P}};
  matmul(h_.data(), H, T, outs, 4);
  std::optional<Timer> phase;
  phase.emplace(*this, Phase::DeltaNet);

  const int64_t cb = 64;
  const int64_t nblk = (conv_dim + cb - 1) / cb;
  const int64_t runs = static_cast<int64_t>(batch.runs.size());
  pool_.parallel_for(runs * nblk, 1, [&](int64_t i0, int64_t i1) {
    for (int64_t item = i0; item < i1; ++item) {
      const BatchRun& run = batch.runs[static_cast<size_t>(item / nblk)];
      const int64_t c0 = (item % nblk) * cb;
      const int64_t c1 = std::min(c0 + cb, conv_dim);
      float* st = conv_state(run.seq->slot, li);
      for (int64_t i = 0; i < run.rows; ++i) {
        const int64_t t = run.first_row + i;
        const float* in = proj + t * P;
        float* out = conv + t * conv_dim;
        for (int64_t ch = c0; ch < c1; ++ch) {
          const float* w = L.lin.conv.data() + ch * ck;
          float acc = 0.0f;
          for (int64_t j = 0; j + 1 < ck; ++j) acc += w[j] * st[j * conv_dim + ch];
          acc += w[ck - 1] * in[ch];
          out[ch] = acc;
          for (int64_t j = 0; j + 2 < ck; ++j) st[j * conv_dim + ch] = st[(j + 1) * conv_dim + ch];
          st[(ck - 2) * conv_dim + ch] = in[ch];
        }
        K.silu(out + c0, c1 - c0);
        if (i < run.snapshots) {
          float* snap = conv_snapshot(run.seq->slot, i, li);
          for (int64_t j = 0; j + 1 < ck; ++j) {
            std::memcpy(snap + j * conv_dim + c0, st + j * conv_dim + c0,
                        static_cast<size_t>(c1 - c0) * 4);
          }
        }
      }
    }
  });

  const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kd)));
  pool_.parallel_for(T * kh, 16, [&](int64_t i0, int64_t i1) {
    for (int64_t item = i0; item < i1; ++item) {
      int64_t t = item / kh;
      int64_t h = item % kh;
      l2_normalize(conv + t * conv_dim + h * kd, kd, qscale);
      l2_normalize(conv + t * conv_dim + key_dim + h * kd, kd, 1.0f);
    }
  });
  float* gates = gates_.data();
  for (int64_t t = 0; t < T; ++t) {
    const float* b = proj + t * P + conv_dim + value_dim;
    const float* a = b + vh;
    for (int64_t h = 0; h < vh; ++h) {
      float beta = 1.0f / (1.0f + std::exp(-b[h]));
      float g = L.lin.neg_exp_a[static_cast<size_t>(h)] *
                softplus(a[h] + L.lin.dt_bias[static_cast<size_t>(h)]);
      gates[(t * vh + h) * 2] = beta;
      gates[(t * vh + h) * 2 + 1] = std::exp(g);
    }
  }
  const int64_t ratio = vh / kh;
  const int64_t w = ssm_block_cols_;
  const int64_t nb = vd / w;
  const int64_t block = kd * w;
  float* core = attn_.data();
  pool_.parallel_for(runs * vh * nb, 1, [&](int64_t i0, int64_t i1) {
    float* scratch = thread_buffer(static_cast<size_t>(w));
    for (int64_t item = i0; item < i1; ++item) {
      const BatchRun& run = batch.runs[static_cast<size_t>(item / (vh * nb))];
      const int64_t h = item / nb % vh;
      const int64_t b = item % nb;
      const int64_t hk = h / ratio;
      const int64_t col = h * vd + b * w;
      float* S = ssm_state(run.seq->slot, li, h) + b * block;
      if (run.snapshots == 0) {
        DeltaSeq seq;
        seq.tokens = run.rows;
        const float* row = conv + run.first_row * conv_dim;
        seq.q = row + hk * kd;
        seq.k = row + key_dim + hk * kd;
        seq.v = row + 2 * key_dim + col;
        seq.qkv_stride = conv_dim;
        seq.gates = gates + (run.first_row * vh + h) * 2;
        seq.gate_stride = vh * 2;
        seq.out = core + run.first_row * value_dim + col;
        seq.out_stride = value_dim;
        K.deltanet_seq(S, seq, scratch, kd, w);
        continue;
      }
      for (int64_t i = 0; i < run.rows; ++i) {
        const int64_t t = run.first_row + i;
        const float* row = conv + t * conv_dim;
        K.deltanet_step(S, row + hk * kd, row + key_dim + hk * kd, row + 2 * key_dim + col,
                        gates[(t * vh + h) * 2 + 1], gates[(t * vh + h) * 2],
                        core + t * value_dim + col, scratch, kd, w);
        if (i < run.snapshots) {
          std::memcpy(ssm_snapshot(run.seq->slot, i, li, h) + b * block, S,
                      static_cast<size_t>(block) * 4);
        }
      }
    }
  });
  const float eps = c.rms_eps;
  pool_.parallel_for(T * vh, 16, [&](int64_t i0, int64_t i1) {
    for (int64_t item = i0; item < i1; ++item) {
      int64_t t = item / vh;
      int64_t h = item % vh;
      float* o = core + t * value_dim + h * vd;
      rms_norm(o, o, L.lin.norm.data(), vd, eps);
      K.silu_mul(proj + t * P + conv_dim + h * vd, o, o, vd);
    }
  });
  phase.reset();
  Projection out = {&L.lin.out, tmp_.data(), H};
  matmul(core, value_dim, T, &out, 1);
  add_rows(x_.data(), tmp_.data(), T);
}

void Runtime::mlp_block(const LayerWeights& L, int64_t rows) {
  const KernelSet& K = kernels();
  const int64_t H = cfg_.hidden;
  const int64_t I = cfg_.intermediate;
  const int64_t P = proj_stride_;
  float* proj = proj_.data();
  Projection outs[2] = {{&L.gate, proj, P}, {&L.up, proj + I, P}};
  matmul(h_.data(), H, rows, outs, 2);
  {
    Timer timer(*this, Phase::Activations);
    pool_.parallel_for(rows, 4, [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) {
        K.silu_mul(proj + t * P, proj + t * P + I, act_.data() + t * I, I);
      }
    });
  }
  Projection down = {&L.down, tmp_.data(), H};
  matmul(act_.data(), I, rows, &down, 1);
  add_rows(x_.data(), tmp_.data(), rows);
}

void Runtime::decoder_layer(const LayerWeights& L, const Batch& batch, bool mtp) {
  const int64_t T = batch.rows();
  norm_rows(x_.data(), h_.data(), L.input_norm, T);
  if (L.linear) {
    linear_block(L, batch);
  } else {
    attention_block(L, batch, mtp);
  }
  norm_rows(x_.data(), h_.data(), L.post_norm, T);
  mlp_block(L, T);
}

void Runtime::forward(const Batch& batch, float* hidden_out) {
  prepare(batch);
  for (const BatchRun& run : batch.runs) {
    int64_t first = batch.positions[static_cast<size_t>(run.first_row)];
    if (first != run.seq->length) fail("batch positions must continue the sequence");
    if (static_cast<int64_t>(run.seq->blocks.size()) * opt_.block_size < first + run.rows) {
      fail("kv cache blocks were not reserved");
    }
  }
  embed_rows(batch);
  for (const LayerWeights& L : model_.weights().layers) decoder_layer(L, batch, false);
  const int64_t H = cfg_.hidden;
  const std::vector<float>& w = model_.weights().final_norm;
  const int64_t n = static_cast<int64_t>(batch.out_rows.size());
  {
    Timer timer(*this, Phase::Norms);
    pool_.parallel_for(n, 4, [&](int64_t k0, int64_t k1) {
      for (int64_t k = k0; k < k1; ++k) {
        rms_norm(x_.data() + batch.out_rows[static_cast<size_t>(k)] * H, hidden_out + k * H,
                 w.data(), H, cfg_.rms_eps);
      }
    });
  }
  for (const BatchRun& run : batch.runs) run.seq->length += run.rows;
}

void Runtime::forward_mtp(const Batch& batch, const float* hidden_in, float* hidden_out) {
  Timer timer(*this, Phase::Mtp);
  const MtpWeights& mtp = model_.weights().mtp;
  if (!mtp.present) fail("model has no multi-token prediction head");
  prepare(batch);
  for (const BatchRun& run : batch.runs) {
    int64_t first = batch.positions[static_cast<size_t>(run.first_row)];
    if (first != run.seq->mtp_length) fail("mtp positions must continue the sequence");
    if (static_cast<int64_t>(run.seq->mtp_blocks.size()) * opt_.block_size < first + run.rows) {
      fail("mtp kv cache blocks were not reserved");
    }
  }
  const int64_t T = batch.rows();
  const int64_t H = cfg_.hidden;
  const WeightView& e = model_.weights().embed;
  pool_.parallel_for(T, 4, [&](int64_t t0, int64_t t1) {
    float* row = thread_buffer(static_cast<size_t>(H));
    for (int64_t t = t0; t < t1; ++t) {
      dequantize_row(e, batch.tokens[static_cast<size_t>(t)], row);
      float* dst = mtp_in_.data() + t * 2 * H;
      rms_norm(row, dst, mtp.norm_embed.data(), H, cfg_.rms_eps);
      rms_norm(hidden_in + t * H, dst + H, mtp.norm_hidden.data(), H, cfg_.rms_eps);
    }
  });
  Projection fc = {&mtp.fc, x_.data(), H};
  matmul(mtp_in_.data(), 2 * H, T, &fc, 1);
  decoder_layer(mtp.layer, batch, true);
  const int64_t n = static_cast<int64_t>(batch.out_rows.size());
  pool_.parallel_for(n, 4, [&](int64_t k0, int64_t k1) {
    for (int64_t k = k0; k < k1; ++k) {
      rms_norm(x_.data() + batch.out_rows[static_cast<size_t>(k)] * H, hidden_out + k * H,
               mtp.norm.data(), H, cfg_.rms_eps);
    }
  });
  for (const BatchRun& run : batch.runs) run.seq->mtp_length += run.rows;
}

void Runtime::logits(const float* hidden, int64_t rows, float* out) {
  logits_subset(hidden, rows, model_.weights().lm_head, out);
}

void Runtime::logits_subset(const float* hidden, int64_t rows, const WeightView& head, float* out,
                            Phase phase) {
  Timer timer(*this, phase);
  Projection p = {&head, out, head.rows};
  matmul(hidden, cfg_.hidden, rows, &p, 1);
}

}
