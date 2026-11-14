#include "model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <set>
#include <unordered_map>

#include "common.h"
#include "thread_pool.h"

namespace infer {

static const Json& text_config(const Json& doc) {
  const Json* t = doc.find("text_config");
  return t != nullptr && t->is_object() ? *t : doc;
}

ModelConfig ModelConfig::from_json(const Json& doc) {
  const Json& c = text_config(doc);
  std::string type = c.get_string("model_type", doc.get_string("model_type"));
  ModelConfig m;
  if (type == "qwen3") {
    m.arch = Arch::Qwen3;
  } else if (type == "qwen3_5" || type == "qwen3_5_text") {
    m.arch = Arch::Qwen35;
  } else {
    fail("unsupported model_type '" + type + "' (supported: qwen3, qwen3_5)");
  }
  m.vocab = c.get_int("vocab_size", 0);
  m.hidden = c.get_int("hidden_size", 0);
  m.intermediate = c.get_int("intermediate_size", 0);
  m.layers = c.get_int("num_hidden_layers", 0);
  m.heads = c.get_int("num_attention_heads", 0);
  m.kv_heads = c.get_int("num_key_value_heads", m.heads);
  m.head_dim = c.get_int("head_dim", m.heads > 0 ? m.hidden / m.heads : 0);
  m.rms_eps = static_cast<float>(c.get_number("rms_norm_eps", 1e-6));
  m.max_position = c.get_int("max_position_embeddings", 32768);
  m.tie_embeddings = c.get_bool("tie_word_embeddings", doc.get_bool("tie_word_embeddings", false));
  if (c.get_bool("attention_bias", false)) fail("attention bias is not supported");
  if (c.get_string("hidden_act", "silu") != "silu") fail("only silu activations are supported");
  const Json& rope = c["rope_parameters"];
  double theta = c.get_number("rope_theta", 10000.0);
  double partial = c.get_number("partial_rotary_factor", 1.0);
  if (rope.is_object()) {
    theta = rope.get_number("rope_theta", theta);
    partial = rope.get_number("partial_rotary_factor", partial);
    std::string rope_type = rope.get_string("rope_type", "default");
    if (rope_type != "default") fail("unsupported rope_type '" + rope_type + "'");
  }
  if (c["rope_scaling"].is_object()) fail("rope scaling is not supported");
  m.rope_theta = static_cast<float>(theta);
  m.rotary_dim = static_cast<int64_t>(static_cast<double>(m.head_dim) * partial);
  m.linear.assign(static_cast<size_t>(std::max<int64_t>(m.layers, 0)), 0);
  if (m.arch == Arch::Qwen35) {
    m.attn_gate = c.get_bool("attn_output_gate", true);
    if (!m.attn_gate) fail("qwen3_5 without the attention output gate is not supported");
    m.zero_centered_norm = true;
    const Json& types = c["layer_types"];
    if (types.is_array()) {
      if (static_cast<int64_t>(types.size()) < m.layers)
        fail("layer_types is shorter than num_hidden_layers");
      for (int64_t i = 0; i < m.layers; ++i) {
        std::string t = types.at(static_cast<size_t>(i)).as_string();
        if (t != "linear_attention" && t != "full_attention") fail("unknown layer type " + t);
        m.linear[static_cast<size_t>(i)] = t == "linear_attention";
      }
    } else {
      int64_t interval = c.get_int("full_attention_interval", 4);
      for (int64_t i = 0; i < m.layers; ++i)
        m.linear[static_cast<size_t>(i)] = (i + 1) % interval != 0;
    }
    m.lin_k_heads = c.get_int("linear_num_key_heads", 16);
    m.lin_v_heads = c.get_int("linear_num_value_heads", 32);
    m.lin_k_dim = c.get_int("linear_key_head_dim", 128);
    m.lin_v_dim = c.get_int("linear_value_head_dim", 128);
    m.conv_kernel = c.get_int("linear_conv_kernel_dim", 4);
    m.mtp_layers = c.get_int("mtp_num_hidden_layers", 0);
    const Json& mlp_only = c["mlp_only_layers"];
    if (mlp_only.is_array() && mlp_only.size() > 0) fail("mlp_only_layers is not supported");
  }
  m.validate();
  return m;
}

int64_t ModelConfig::attn_layers() const {
  int64_t n = 0;
  for (uint8_t l : linear) n += l ? 0 : 1;
  return n;
}

int64_t ModelConfig::linear_layers() const {
  int64_t n = 0;
  for (uint8_t l : linear) n += l ? 1 : 0;
  return n;
}

void ModelConfig::validate() const {
  if (vocab <= 0 || hidden <= 0 || intermediate <= 0 || layers <= 0 || heads <= 0 ||
      kv_heads <= 0 || head_dim <= 0) {
    fail("model config has missing or invalid sizes");
  }
  if (heads % kv_heads != 0) fail("num_attention_heads must be a multiple of num_key_value_heads");
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0)
    fail("invalid rotary dimension");
  if (hidden % kQBlock != 0 || intermediate % kQBlock != 0 || (heads * head_dim) % kQBlock != 0) {
    fail("model dimensions must be multiples of 32");
  }
  if (linear_layers() > 0) {
    if (lin_k_heads <= 0 || lin_v_heads <= 0 || lin_k_dim <= 0 || lin_v_dim <= 0 ||
        conv_kernel < 2) {
      fail("invalid linear attention sizes");
    }
    if (lin_v_heads % lin_k_heads != 0) fail("linear value heads must be a multiple of key heads");
    if ((lin_v_heads * lin_v_dim) % kQBlock != 0)
      fail("linear value size must be a multiple of 32");
  }
  if (attn_layers() == 0) fail("model needs at least one full attention layer");
}

class TensorSource {
 public:
  void add(std::unique_ptr<SafeTensors> file) {
    for (const TensorInfo& t : file->tensors()) index_[t.name] = &t;
    files_.push_back(std::move(file));
  }
  const TensorInfo* find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : it->second;
  }
  const TensorInfo& get(const std::string& name) const {
    const TensorInfo* t = find(name);
    if (t == nullptr) fail("missing tensor " + name);
    return *t;
  }
  std::vector<std::unique_ptr<SafeTensors>>& files() { return files_; }

 private:
  std::vector<std::unique_ptr<SafeTensors>> files_;
  std::unordered_map<std::string, const TensorInfo*> index_;
};

static float element_f32(const TensorInfo& t, int64_t i) {
  switch (t.dtype) {
    case DType::F32: {
      float v;
      std::memcpy(&v, t.data + i * 4, 4);
      return v;
    }
    case DType::BF16: {
      uint16_t v;
      std::memcpy(&v, t.data + i * 2, 2);
      return bf16_to_f32(v);
    }
    case DType::F16: {
      uint16_t v;
      std::memcpy(&v, t.data + i * 2, 2);
      return f16_to_f32(v);
    }
    default:
      fail("tensor " + t.name + " is not a float tensor");
  }
}

static std::vector<float> load_vector(const TensorSource& src, const std::string& name, int64_t n) {
  const TensorInfo& t = src.get(name);
  if (t.numel() != n)
    fail(strf("tensor %s has %lld elements, expected %lld", name.c_str(),
              static_cast<long long>(t.numel()), static_cast<long long>(n)));
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = element_f32(t, i);
  return v;
}

static WeightView load_matrix(const TensorSource& src, const std::string& name, int64_t rows,
                              int64_t cols) {
  const TensorInfo& t = src.get(name);
  WeightView w;
  w.rows = rows;
  w.cols = cols;
  w.data = t.data;
  auto expect = [&](int64_t r, int64_t c) {
    if (t.shape.size() != 2 || t.shape[0] != r || t.shape[1] != c) {
      fail(strf("tensor %s has an unexpected shape, expected [%lld, %lld]", name.c_str(),
                static_cast<long long>(r), static_cast<long long>(c)));
    }
  };
  switch (t.dtype) {
    case DType::F32:
      w.type = WType::F32;
      expect(rows, cols);
      return w;
    case DType::F16:
      w.type = WType::F16;
      expect(rows, cols);
      return w;
    case DType::BF16:
      w.type = WType::BF16;
      expect(rows, cols);
      return w;
    case DType::I8:
    case DType::U8: {
      w.type = t.dtype == DType::I8 ? WType::Q8 : WType::Q4;
      const int64_t stored = stored_rows(w.type, rows);
      expect(stored, w.type == WType::Q8 ? cols : cols / 2);
      const TensorInfo& s = src.get(name + ".scales");
      if (s.dtype != DType::F16 || s.numel() != stored * (cols / kQBlock))
        fail("bad scales for " + name);
      w.scales = reinterpret_cast<const uint16_t*>(s.data);
      if (reinterpret_cast<uintptr_t>(w.scales) % 2 != 0) fail("misaligned scales for " + name);
      return w;
    }
    default:
      fail("unsupported dtype for weight " + name);
  }
}

static void zero_center(std::vector<float>& v) {
  for (float& x : v) x = 1.0f + x;
}

static void bind_layer(const TensorSource& src, const ModelConfig& c, const std::string& p,
                       bool linear, LayerWeights& L) {
  const int64_t H = c.hidden;
  L.linear = linear;
  L.input_norm = load_vector(src, p + "input_layernorm.weight", H);
  L.post_norm = load_vector(src, p + "post_attention_layernorm.weight", H);
  if (c.zero_centered_norm) {
    zero_center(L.input_norm);
    zero_center(L.post_norm);
  }
  if (linear) {
    const int64_t key_dim = c.lin_k_heads * c.lin_k_dim;
    const int64_t value_dim = c.lin_v_heads * c.lin_v_dim;
    const std::string a = p + "linear_attn.";
    L.lin.qkv = load_matrix(src, a + "in_proj_qkv.weight", 2 * key_dim + value_dim, H);
    L.lin.z = load_matrix(src, a + "in_proj_z.weight", value_dim, H);
    L.lin.b = load_matrix(src, a + "in_proj_b.weight", c.lin_v_heads, H);
    L.lin.a = load_matrix(src, a + "in_proj_a.weight", c.lin_v_heads, H);
    L.lin.out = load_matrix(src, a + "out_proj.weight", H, value_dim);
    L.lin.conv = load_vector(src, a + "conv1d.weight", c.conv_dim() * c.conv_kernel);
    std::vector<float> a_log = load_vector(src, a + "A_log", c.lin_v_heads);
    L.lin.neg_exp_a.resize(a_log.size());
    for (size_t i = 0; i < a_log.size(); ++i) L.lin.neg_exp_a[i] = -std::exp(a_log[i]);
    L.lin.dt_bias = load_vector(src, a + "dt_bias", c.lin_v_heads);
    L.lin.norm = load_vector(src, a + "norm.weight", c.lin_v_dim);
  } else {
    const std::string a = p + "self_attn.";
    L.attn.q = load_matrix(src, a + "q_proj.weight", c.q_rows(), H);
    L.attn.k = load_matrix(src, a + "k_proj.weight", c.kv_heads * c.head_dim, H);
    L.attn.v = load_matrix(src, a + "v_proj.weight", c.kv_heads * c.head_dim, H);
    L.attn.o = load_matrix(src, a + "o_proj.weight", H, c.heads * c.head_dim);
    L.attn.q_norm = load_vector(src, a + "q_norm.weight", c.head_dim);
    L.attn.k_norm = load_vector(src, a + "k_norm.weight", c.head_dim);
    if (c.zero_centered_norm) {
      zero_center(L.attn.q_norm);
      zero_center(L.attn.k_norm);
    }
  }
  L.gate = load_matrix(src, p + "mlp.gate_proj.weight", c.intermediate, H);
  L.up = load_matrix(src, p + "mlp.up_proj.weight", c.intermediate, H);
  L.down = load_matrix(src, p + "mlp.down_proj.weight", H, c.intermediate);
}

static std::string model_prefix(const TensorSource& src) {
  if (src.find("model.language_model.embed_tokens.weight") != nullptr)
    return "model.language_model.";
  if (src.find("model.embed_tokens.weight") != nullptr) return "model.";
  fail("cannot find the token embedding tensor");
}

void Model::bind(TensorSource& src) {
  const ModelConfig& c = config_;
  const std::string p = model_prefix(src);
  weights_.embed = load_matrix(src, p + "embed_tokens.weight", c.vocab, c.hidden);
  weights_.final_norm = load_vector(src, p + "norm.weight", c.hidden);
  if (c.zero_centered_norm) zero_center(weights_.final_norm);
  if (c.tie_embeddings || src.find("lm_head.weight") == nullptr) {
    if (!c.tie_embeddings) fail("missing lm_head.weight");
    weights_.lm_head = weights_.embed;
  } else {
    weights_.lm_head = load_matrix(src, "lm_head.weight", c.vocab, c.hidden);
  }
  weights_.layers.resize(static_cast<size_t>(c.layers));
  int64_t attn_index = 0;
  int64_t linear_index = 0;
  for (int64_t i = 0; i < c.layers; ++i) {
    LayerWeights& L = weights_.layers[static_cast<size_t>(i)];
    bool linear = c.linear[static_cast<size_t>(i)] != 0;
    bind_layer(src, c, p + "layers." + std::to_string(i) + ".", linear, L);
    if (linear) {
      L.linear_index = linear_index++;
    } else {
      L.attn_index = attn_index++;
    }
  }
  MtpWeights& mtp = weights_.mtp;
  if (c.mtp_layers > 0 && src.find("mtp.fc.weight") != nullptr) {
    mtp.present = true;
    mtp.fc = load_matrix(src, "mtp.fc.weight", c.hidden, 2 * c.hidden);
    mtp.norm_embed = load_vector(src, "mtp.pre_fc_norm_embedding.weight", c.hidden);
    mtp.norm_hidden = load_vector(src, "mtp.pre_fc_norm_hidden.weight", c.hidden);
    mtp.norm = load_vector(src, "mtp.norm.weight", c.hidden);
    zero_center(mtp.norm_embed);
    zero_center(mtp.norm_hidden);
    zero_center(mtp.norm);
    bind_layer(src, c, "mtp.layers.0.", false, mtp.layer);
    mtp.layer.attn_index = 0;
  }
  files_ = std::move(src.files());
}

void Model::finish() {
  WType t = weights_.layers.front().gate.type;
  weight_label_ = wtype_name(t);
}

static std::string model_name(const std::string& path, bool directory) {
  std::filesystem::path p(
      std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size()));
  if (!p.has_filename()) p = p.parent_path();
  if (!directory && p.extension() == ".infer") p = p.stem();
  std::u8string s = p.filename().u8string();
  return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

std::unique_ptr<Model> Model::load(const std::string& path) {
  std::unique_ptr<Model> m(new Model());
  TensorSource src;
  const bool directory = is_directory(path);
  if (directory) {
    m->config_text_ = read_text_file(join_path(path, "config.json"));
    std::string index_path = join_path(path, "model.safetensors.index.json");
    if (path_exists(index_path)) {
      Json index = Json::parse(read_text_file(index_path));
      std::set<std::string> shards;
      for (const Json& f : index["weight_map"].items()) shards.insert(f.as_string());
      for (const std::string& f : shards)
        src.add(std::make_unique<SafeTensors>(join_path(path, f)));
    } else {
      src.add(std::make_unique<SafeTensors>(join_path(path, "model.safetensors")));
    }
    m->tokenizer_ = Tokenizer::from_json(read_text_file(join_path(path, "tokenizer.json")));
  } else {
    auto file = std::make_unique<SafeTensors>(path);
    const std::string format = file->metadata("infer.format");
    if (format.empty()) fail(path + " is not a model file written by infer convert");
    if (format != "2") fail(path + " uses an older layout, convert the model again");
    m->config_text_ = file->metadata("infer.config");
    const TensorInfo* tok = file->find("tokenizer.json");
    if (tok == nullptr) fail(path + " has no tokenizer");
    m->tokenizer_ = Tokenizer::from_json(std::string_view(reinterpret_cast<const char*>(tok->data),
                                                          static_cast<size_t>(tok->nbytes)));
    src.add(std::move(file));
  }
  m->config_ = ModelConfig::from_json(Json::parse(m->config_text_));
  m->name_ = model_name(path, directory);
  m->bind(src);
  m->finish();
  return m;
}

std::unique_ptr<Model> Model::random(const ModelConfig& config, WType type, uint64_t seed,
                                     std::unique_ptr<Tokenizer> tokenizer, bool with_mtp) {
  config.validate();
  std::unique_ptr<Model> m(new Model());
  m->config_ = config;
  if (!with_mtp) m->config_.mtp_layers = 0;
  if (with_mtp && m->config_.mtp_layers == 0) m->config_.mtp_layers = 1;
  m->tokenizer_ = std::move(tokenizer);
  m->name_ = "random";
  const ModelConfig& c = m->config_;
  Rng rng(seed);
  auto matrix = [&](int64_t rows, int64_t cols, float scale) {
    std::vector<float> v(static_cast<size_t>(rows * cols));
    for (float& x : v) x = rng.normal() * scale;
    auto w = std::make_unique<OwnedWeight>();
    make_weight(*w, v.data(), rows, cols, type);
    WeightView view = w->view;
    m->owned_.push_back(std::move(w));
    return view;
  };
  auto norm = [&](int64_t n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (float& x : v) x = 1.0f + 0.1f * rng.normal();
    return v;
  };
  const int64_t H = c.hidden;
  auto layer = [&](LayerWeights& L, bool linear) {
    L.linear = linear;
    L.input_norm = norm(H);
    L.post_norm = norm(H);
    if (linear) {
      int64_t key_dim = c.lin_k_heads * c.lin_k_dim;
      int64_t value_dim = c.lin_v_heads * c.lin_v_dim;
      L.lin.qkv = matrix(2 * key_dim + value_dim, H, 0.05f);
      L.lin.z = matrix(value_dim, H, 0.05f);
      L.lin.b = matrix(c.lin_v_heads, H, 0.05f);
      L.lin.a = matrix(c.lin_v_heads, H, 0.05f);
      L.lin.out = matrix(H, value_dim, 0.05f);
      L.lin.conv.resize(static_cast<size_t>(c.conv_dim() * c.conv_kernel));
      for (float& x : L.lin.conv) x = 0.3f * rng.normal();
      for (int64_t h = 0; h < c.lin_v_heads; ++h) {
        L.lin.neg_exp_a.push_back(-(1.0f + 15.0f * rng.uniform()));
        L.lin.dt_bias.push_back(0.5f * rng.normal());
      }
      L.lin.norm = norm(c.lin_v_dim);
    } else {
      L.attn.q = matrix(c.q_rows(), H, 0.05f);
      L.attn.k = matrix(c.kv_heads * c.head_dim, H, 0.05f);
      L.attn.v = matrix(c.kv_heads * c.head_dim, H, 0.05f);
      L.attn.o = matrix(H, c.heads * c.head_dim, 0.05f);
      L.attn.q_norm = norm(c.head_dim);
      L.attn.k_norm = norm(c.head_dim);
    }
    L.gate = matrix(c.intermediate, H, 0.05f);
    L.up = matrix(c.intermediate, H, 0.05f);
    L.down = matrix(H, c.intermediate, 0.05f);
  };
  ModelWeights& w = m->weights_;
  w.embed = matrix(c.vocab, H, 0.5f);
  w.lm_head = c.tie_embeddings ? w.embed : matrix(c.vocab, H, 0.05f);
  w.final_norm = norm(H);
  w.layers.resize(static_cast<size_t>(c.layers));
  int64_t attn_index = 0;
  int64_t linear_index = 0;
  for (int64_t i = 0; i < c.layers; ++i) {
    bool linear = c.linear[static_cast<size_t>(i)] != 0;
    LayerWeights& L = w.layers[static_cast<size_t>(i)];
    layer(L, linear);
    if (linear) {
      L.linear_index = linear_index++;
    } else {
      L.attn_index = attn_index++;
    }
  }
  if (with_mtp) {
    w.mtp.present = true;
    w.mtp.fc = matrix(H, 2 * H, 0.05f);
    w.mtp.norm_embed = norm(H);
    w.mtp.norm_hidden = norm(H);
    w.mtp.norm = norm(H);
    layer(w.mtp.layer, false);
    w.mtp.layer.attn_index = 0;
  }
  m->finish();
  return m;
}

std::vector<int32_t> Model::stop_tokens() const {
  std::vector<int32_t> out;
  for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
    int32_t id = tokenizer_->find_added(s);
    if (id >= 0) out.push_back(id);
  }
  return out;
}

size_t Model::decode_bytes() const {
  size_t total = weights_.lm_head.bytes();
  for (const LayerWeights& L : weights_.layers) {
    if (L.linear) {
      total += L.lin.qkv.bytes() + L.lin.z.bytes() + L.lin.b.bytes() + L.lin.a.bytes() +
               L.lin.out.bytes();
    } else {
      total += L.attn.q.bytes() + L.attn.k.bytes() + L.attn.v.bytes() + L.attn.o.bytes();
    }
    total += L.gate.bytes() + L.up.bytes() + L.down.bytes();
  }
  return total;
}

size_t Model::parameter_count() const {
  auto n = [](const WeightView& w) { return static_cast<size_t>(w.rows * w.cols); };
  size_t total = n(weights_.embed) + weights_.final_norm.size();
  if (weights_.lm_head.data != weights_.embed.data) total += n(weights_.lm_head);
  for (const LayerWeights& L : weights_.layers) {
    total += L.input_norm.size() + L.post_norm.size() + n(L.gate) + n(L.up) + n(L.down);
    if (L.linear) {
      total += n(L.lin.qkv) + n(L.lin.z) + n(L.lin.b) + n(L.lin.a) + n(L.lin.out) +
               L.lin.conv.size() + L.lin.neg_exp_a.size() + L.lin.dt_bias.size() +
               L.lin.norm.size();
    } else {
      total += n(L.attn.q) + n(L.attn.k) + n(L.attn.v) + n(L.attn.o) + L.attn.q_norm.size() +
               L.attn.k_norm.size();
    }
  }
  return total;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool writes_residual(const std::string& name) {
  return ends_with(name, "down_proj.weight") || ends_with(name, "o_proj.weight") ||
         ends_with(name, "out_proj.weight");
}

static bool is_matrix_name(const std::string& name) {
  static const char* const suffixes[] = {
      "_proj.weight",     "embed_tokens.weight", "lm_head.weight",   "in_proj_qkv.weight",
      "in_proj_z.weight", "in_proj_b.weight",    "in_proj_a.weight", "mtp.fc.weight"};
  for (const char* s : suffixes) {
    std::string suffix(s);
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return true;
    }
  }
  return false;
}

void convert_model(const std::string& hf_dir, const std::string& out_path,
                   const ConvertOptions& options) {
  if (!is_directory(hf_dir)) fail(hf_dir + " is not a Hugging Face model directory");
  std::string config_text = read_text_file(join_path(hf_dir, "config.json"));
  ModelConfig config = ModelConfig::from_json(Json::parse(config_text));
  std::string tokenizer_text = read_text_file(join_path(hf_dir, "tokenizer.json"));
  Tokenizer::from_json(tokenizer_text);
  TensorSource src;
  std::string index_path = join_path(hf_dir, "model.safetensors.index.json");
  if (path_exists(index_path)) {
    Json index = Json::parse(read_text_file(index_path));
    std::set<std::string> shards;
    for (const Json& f : index["weight_map"].items()) shards.insert(f.as_string());
    for (const std::string& f : shards)
      src.add(std::make_unique<SafeTensors>(join_path(hf_dir, f)));
  } else {
    src.add(std::make_unique<SafeTensors>(join_path(hf_dir, "model.safetensors")));
  }
  std::vector<const TensorInfo*> tensors;
  for (const auto& f : src.files()) {
    for (const TensorInfo& t : f->tensors()) {
      if (t.name.rfind("model.visual.", 0) == 0 || t.name.rfind("__pad.", 0) == 0) continue;
      if (t.name.rfind("mtp.", 0) == 0 && config.mtp_layers == 0) continue;
      if (t.name == "lm_head.weight" && config.tie_embeddings) continue;
      tensors.push_back(&t);
    }
  }
  std::sort(tensors.begin(), tensors.end(),
            [](const TensorInfo* a, const TensorInfo* b) { return a->name < b->name; });
  ThreadPool pool(std::max(1, options.threads));
  std::vector<TensorSpec> specs;
  for (const TensorInfo* t : tensors) {
    bool matrix = t->shape.size() == 2 && is_matrix_name(t->name);
    WType type = options.type;
    if (!matrix) {
      TensorSpec s;
      s.name = t->name;
      s.dtype = DType::F32;
      s.shape = t->shape;
      s.fill = [t](uint8_t* dst, uint64_t) {
        for (int64_t i = 0; i < t->numel(); ++i) {
          float v = element_f32(*t, i);
          std::memcpy(dst + i * 4, &v, 4);
        }
      };
      specs.push_back(std::move(s));
      continue;
    }
    const int64_t rows = t->shape[0];
    const int64_t cols = t->shape[1];
    bool embedding = t->name.find("embed_tokens") != std::string::npos ||
                     t->name.find("lm_head") != std::string::npos;
    if (type == WType::Q4 && (embedding || writes_residual(t->name))) type = WType::Q8;
    if (wtype_quantized(type) && (rows < 64 || cols % kQBlock != 0)) type = WType::BF16;
    if (!wtype_quantized(type)) {
      TensorSpec s;
      s.name = t->name;
      s.dtype = type == WType::F32 ? DType::F32 : type == WType::F16 ? DType::F16 : DType::BF16;
      s.shape = t->shape;
      s.fill = [t, type, cols, &pool](uint8_t* dst, uint64_t) {
        const size_t row_bytes = wtype_row_bytes(type, cols);
        pool.parallel_for(t->shape[0], 64, [&](int64_t r0, int64_t r1) {
          std::vector<float> row(static_cast<size_t>(cols));
          for (int64_t r = r0; r < r1; ++r) {
            for (int64_t c = 0; c < cols; ++c)
              row[static_cast<size_t>(c)] = element_f32(*t, r * cols + c);
            encode_row(row.data(), cols, type, dst + r * row_bytes);
          }
        });
      };
      specs.push_back(std::move(s));
      continue;
    }
    auto scales = std::make_shared<std::vector<uint16_t>>();
    TensorSpec data;
    data.name = t->name;
    data.dtype = type == WType::Q8 ? DType::I8 : DType::U8;
    const int64_t stored = stored_rows(type, rows);
    data.shape = {stored, type == WType::Q8 ? cols : cols / 2};
    data.fill = [t, type, rows, cols, stored, scales, &pool](uint8_t* dst, uint64_t) {
      const int64_t nb = cols / kQBlock;
      scales->assign(static_cast<size_t>(stored * nb), 0);
      init_quant_storage(type, rows, cols, dst, scales->data());
      pool.parallel_for(stored / kRowGroup, 1, [&](int64_t g0, int64_t g1) {
        std::vector<float> row(static_cast<size_t>(cols));
        std::vector<int8_t> q(static_cast<size_t>(cols));
        std::vector<uint16_t> s(static_cast<size_t>(nb));
        for (int64_t r = g0 * kRowGroup; r < std::min(g1 * kRowGroup, rows); ++r) {
          for (int64_t c = 0; c < cols; ++c)
            row[static_cast<size_t>(c)] = element_f32(*t, r * cols + c);
          if (type == WType::Q8) {
            quantize_q8_row(row.data(), cols, q.data(), s.data());
          } else {
            quantize_q4_row(row.data(), cols, q.data(), s.data());
          }
          store_quant_row(type, r, cols, q.data(), s.data(), dst, scales->data());
        }
      });
    };
    specs.push_back(std::move(data));
    TensorSpec sc;
    sc.name = t->name + ".scales";
    sc.dtype = DType::F16;
    sc.shape = {stored, cols / kQBlock};
    sc.fill = [scales](uint8_t* dst, uint64_t nbytes) {
      std::memcpy(dst, scales->data(), static_cast<size_t>(nbytes));
      scales->clear();
      scales->shrink_to_fit();
    };
    specs.push_back(std::move(sc));
  }
  TensorSpec tok;
  tok.name = "tokenizer.json";
  tok.dtype = DType::U8;
  tok.shape = {static_cast<int64_t>(tokenizer_text.size())};
  tok.fill = [&tokenizer_text](uint8_t* dst, uint64_t nbytes) {
    std::memcpy(dst, tokenizer_text.data(), static_cast<size_t>(nbytes));
  };
  specs.push_back(std::move(tok));
  std::vector<std::pair<std::string, std::string>> meta = {
      {"infer.format", "2"},
      {"infer.weights", wtype_name(options.type)},
      {"infer.config", config_text}};
  write_safetensors(out_path, meta, specs, 64);
}

}
