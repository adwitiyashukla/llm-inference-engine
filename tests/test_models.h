#pragma once

#include <memory>
#include <string>

#include "json.h"
#include "model.h"
#include "tokenizer.h"

namespace infer {

inline std::string byte_char(int b) {
  int extra = 0;
  for (int i = 0; i < 256; ++i) {
    bool printable = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255);
    uint32_t cp = printable ? static_cast<uint32_t>(i) : static_cast<uint32_t>(256 + extra++);
    if (i == b) {
      std::string s;
      append_utf8(cp, s);
      return s;
    }
  }
  return {};
}

inline std::string tiny_tokenizer_json() {
  Json vocab = Json::object();
  for (int b = 0; b < 256; ++b) vocab.set(byte_char(b), Json(b));
  const char* pairs[][2] = {{"t", "h"},  {"th", "e"}, {"a", "n"},
                            {"an", "d"}, {"i", "n"},  {"\xC4\xA0", "the"}};
  Json merges = Json::array();
  int next = 256;
  for (auto& p : pairs) {
    vocab.set(std::string(p[0]) + p[1], Json(next++));
    merges.push(Json(std::string(p[0]) + " " + p[1]));
  }
  Json added = Json::array();
  const char* specials[] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"};
  for (const char* s : specials) {
    Json t = Json::object();
    t.set("id", Json(next++));
    t.set("content", Json(s));
    t.set("special", Json(true));
    added.push(std::move(t));
  }
  Json model = Json::object();
  model.set("type", Json("BPE"));
  model.set("vocab", std::move(vocab));
  model.set("merges", std::move(merges));
  Json split = Json::object();
  split.set("type", Json("Split"));
  Json pattern = Json::object();
  pattern.set(
      "Regex",
      Json(
          R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"));
  split.set("pattern", std::move(pattern));
  split.set("behavior", Json("Isolated"));
  Json bl = Json::object();
  bl.set("type", Json("ByteLevel"));
  bl.set("use_regex", Json(false));
  Json seq = Json::object();
  seq.set("type", Json("Sequence"));
  Json list = Json::array();
  list.push(std::move(split));
  list.push(std::move(bl));
  seq.set("pretokenizers", std::move(list));
  Json doc = Json::object();
  doc.set("added_tokens", std::move(added));
  Json nfc = Json::object();
  nfc.set("type", Json("NFC"));
  doc.set("normalizer", std::move(nfc));
  doc.set("pre_tokenizer", std::move(seq));
  doc.set("model", std::move(model));
  return doc.dump();
}

inline ModelConfig hybrid_config() {
  ModelConfig c;
  c.arch = Arch::Qwen35;
  c.vocab = 272;
  c.hidden = 64;
  c.intermediate = 128;
  c.layers = 4;
  c.heads = 2;
  c.kv_heads = 1;
  c.head_dim = 32;
  c.rotary_dim = 8;
  c.rope_theta = 10000000.0f;
  c.attn_gate = true;
  c.zero_centered_norm = true;
  c.tie_embeddings = true;
  c.linear = {1, 1, 1, 0};
  c.lin_k_heads = 2;
  c.lin_v_heads = 4;
  c.lin_k_dim = 32;
  c.lin_v_dim = 128;
  c.conv_kernel = 4;
  c.mtp_layers = 1;
  c.max_position = 4096;
  return c;
}

inline ModelConfig dense_config() {
  ModelConfig c;
  c.arch = Arch::Qwen3;
  c.vocab = 272;
  c.hidden = 64;
  c.intermediate = 128;
  c.layers = 2;
  c.heads = 4;
  c.kv_heads = 2;
  c.head_dim = 32;
  c.rotary_dim = 32;
  c.rope_theta = 1000000.0f;
  c.linear = {0, 0};
  c.max_position = 4096;
  return c;
}

inline std::unique_ptr<Model> make_model(const ModelConfig& c, WType type, bool mtp,
                                         uint64_t seed = 7) {
  return Model::random(c, type, seed, Tokenizer::from_json(tiny_tokenizer_json()), mtp);
}

}
