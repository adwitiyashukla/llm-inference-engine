#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infer {

enum class PreTokenizer { Gpt2, Llama3, Qwen2, Qwen35 };

class Tokenizer {
 public:
  static std::unique_ptr<Tokenizer> from_json(std::string_view json_text);

  std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const;
  std::vector<std::string> pre_tokenize(std::string_view text) const;

  const std::string& token_bytes(int32_t id) const;
  int32_t find_added(std::string_view content) const;
  bool is_special(int32_t id) const;
  bool is_added(int32_t id) const;
  int32_t size() const { return static_cast<int32_t>(tokens_.size()); }
  PreTokenizer pre_tokenizer() const { return pre_; }
  bool uses_nfc() const { return nfc_; }

 private:
  Tokenizer() = default;
  void bpe(std::string_view piece, std::vector<int32_t>& out) const;
  void encode_plain(std::string_view text, std::vector<int32_t>& out) const;
  std::vector<std::pair<size_t, size_t>> split(std::string_view text) const;

  struct Merge {
    int32_t rank;
    int32_t id;
  };

  PreTokenizer pre_ = PreTokenizer::Qwen2;
  bool nfc_ = false;
  std::vector<std::string> tokens_;
  std::vector<uint8_t> token_kind_;
  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<uint64_t, Merge> merges_;
  int32_t byte_ids_[256] = {};
  std::unordered_map<std::string, int32_t> added_;
  size_t max_added_len_ = 0;
  bool added_first_byte_[256] = {};
  mutable std::mutex cache_mutex_;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}
  std::string push(int32_t id);
  std::string flush();

 private:
  const Tokenizer& tokenizer_;
  std::string pending_;
};

enum class ChatFormat { Qwen3, Qwen35 };

struct ChatMessage {
  std::string role;
  std::string content;
  std::string reasoning;
};

std::string render_chat(ChatFormat format, const std::vector<ChatMessage>& messages,
                        bool add_generation_prompt, int thinking);

}
