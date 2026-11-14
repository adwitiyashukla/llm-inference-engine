#include "tokenizer.h"

#include <algorithm>
#include <queue>

#include "common.h"
#include "json.h"
#include "unicode.h"

namespace infer {
namespace {

constexpr uint8_t kNormal = 0;
constexpr uint8_t kAdded = 1;
constexpr uint8_t kSpecial = 2;
constexpr uint8_t kUnused = 3;

const char* const kQwen2Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
const char* const kLlama3Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
const char* const kQwen35Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

struct ByteMap {
  uint32_t to_unicode[256];
  int16_t to_byte[512];
  ByteMap() {
    for (auto& v : to_byte) v = -1;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      uint32_t cp = printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
      to_unicode[b] = cp;
      to_byte[cp] = static_cast<int16_t>(b);
    }
  }
};

const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}

bool unmap_bytes(std::string_view mapped, std::string& raw) {
  raw.clear();
  DecodedText d = utf8_decode(mapped);
  const ByteMap& m = byte_map();
  for (uint32_t cp : d.cps) {
    if (cp >= 512 || m.to_byte[cp] < 0) return false;
    raw += static_cast<char>(static_cast<uint8_t>(m.to_byte[cp]));
  }
  return true;
}

uint32_t fold_case(uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') return cp + 32;
  if (cp == 0x17F) return 's';
  return cp;
}

size_t match_contraction(const std::vector<uint32_t>& c, size_t i, bool fold) {
  if (c[i] != '\'' || i + 1 >= c.size()) return 0;
  uint32_t a = fold ? fold_case(c[i + 1]) : c[i + 1];
  if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
  if (i + 2 < c.size()) {
    uint32_t b = fold ? fold_case(c[i + 2]) : c[i + 2];
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
  }
  return 0;
}

std::vector<std::pair<size_t, size_t>> split_llama3(const DecodedText& d, bool marks,
                                                    size_t max_digits) {
  const std::vector<uint32_t>& c = d.cps;
  const size_t n = c.size();
  auto run_char = [marks](uint32_t x) { return is_letter(x) || (marks && is_mark(x)); };
  auto punct = [marks](uint32_t x) {
    return !is_whitespace(x) && !is_letter(x) && !is_number(x) && !(marks && is_mark(x));
  };
  std::vector<std::pair<size_t, size_t>> out;
  size_t i = 0;
  while (i < n) {
    size_t j = i + 1;
    uint32_t x = c[i];
    if (size_t k = match_contraction(c, i, true)) {
      j = i + k;
    } else if (run_char(x)) {
      while (j < n && run_char(c[j])) ++j;
    } else if (x != '\r' && x != '\n' && !is_letter(x) && !is_number(x) && i + 1 < n &&
               run_char(c[i + 1])) {
      j = i + 2;
      while (j < n && run_char(c[j])) ++j;
    } else if (is_number(x)) {
      while (j < n && j - i < max_digits && is_number(c[j])) ++j;
    } else if ((x == ' ' && i + 1 < n && punct(c[i + 1])) || punct(x)) {
      j = x == ' ' ? i + 1 : i;
      while (j < n && punct(c[j])) ++j;
      while (j < n && (c[j] == '\r' || c[j] == '\n')) ++j;
    } else if (is_whitespace(x)) {
      size_t e = i;
      while (e < n && is_whitespace(c[e])) ++e;
      size_t last_newline = n;
      for (size_t p = i; p < e; ++p) {
        if (c[p] == '\r' || c[p] == '\n') last_newline = p;
      }
      if (last_newline != n) {
        j = last_newline + 1;
      } else if (e == n || e - i == 1) {
        j = e;
      } else {
        j = e - 1;
      }
    }
    out.emplace_back(d.offsets[i], d.offsets[j]);
    i = j;
  }
  return out;
}

std::vector<std::pair<size_t, size_t>> split_gpt2(const DecodedText& d) {
  const std::vector<uint32_t>& c = d.cps;
  const size_t n = c.size();
  auto punct = [](uint32_t x) { return !is_whitespace(x) && !is_letter(x) && !is_number(x); };
  auto run = [&](size_t from, auto pred) {
    size_t e = from;
    while (e < n && pred(c[e])) ++e;
    return e;
  };
  auto letter = [](uint32_t x) { return is_letter(x); };
  auto number = [](uint32_t x) { return is_number(x); };
  auto space = [](uint32_t x) { return is_whitespace(x); };
  std::vector<std::pair<size_t, size_t>> out;
  size_t i = 0;
  while (i < n) {
    size_t j = i + 1;
    uint32_t x = c[i];
    bool sp = x == ' ' && i + 1 < n;
    if (size_t k = match_contraction(c, i, false)) {
      j = i + k;
    } else if (sp && is_letter(c[i + 1])) {
      j = run(i + 1, letter);
    } else if (is_letter(x)) {
      j = run(i, letter);
    } else if (sp && is_number(c[i + 1])) {
      j = run(i + 1, number);
    } else if (is_number(x)) {
      j = run(i, number);
    } else if (sp && punct(c[i + 1])) {
      j = run(i + 1, punct);
    } else if (punct(x)) {
      j = run(i, punct);
    } else if (is_whitespace(x)) {
      size_t e = run(i, space);
      j = (e == n || e - i == 1) ? e : e - 1;
    }
    out.emplace_back(d.offsets[i], d.offsets[j]);
    i = j;
  }
  return out;
}

std::string rstrip_newlines(std::string s) {
  while (!s.empty() && s.back() == '\n') s.pop_back();
  return s;
}

std::string lstrip_newlines(std::string s) {
  size_t k = 0;
  while (k < s.size() && s[k] == '\n') ++k;
  return s.substr(k);
}

void split_reasoning(std::string& content, std::string& reasoning) {
  const std::string close = "</think>";
  size_t first = content.find(close);
  if (first == std::string::npos) return;
  std::string head = rstrip_newlines(content.substr(0, first));
  size_t open = head.rfind("<think>");
  reasoning = lstrip_newlines(open == std::string::npos ? head : head.substr(open + 7));
  size_t last = content.rfind(close);
  content = lstrip_newlines(content.substr(last + close.size()));
}

bool is_tool_response(const std::string& s) {
  const std::string a = "<tool_response>";
  const std::string b = "</tool_response>";
  return s.size() >= a.size() && s.compare(0, a.size(), a) == 0 && s.size() >= b.size() &&
         s.compare(s.size() - b.size(), b.size(), b) == 0;
}

}

std::unique_ptr<Tokenizer> Tokenizer::from_json(std::string_view json_text) {
  Json doc = Json::parse(json_text);
  std::unique_ptr<Tokenizer> tok(new Tokenizer());
  const Json& model = doc["model"];
  if (model.get_string("type") != "BPE") fail("tokenizer: only BPE models are supported");
  if (model.get_bool("byte_fallback", false)) fail("tokenizer: byte_fallback is not supported");
  const Json& normalizer = doc["normalizer"];
  if (normalizer.is_object()) {
    if (normalizer.get_string("type") != "NFC") fail("tokenizer: unsupported normalizer");
    tok->nfc_ = true;
  } else if (!normalizer.is_null()) {
    fail("tokenizer: unsupported normalizer");
  }
  const Json& pre = doc["pre_tokenizer"];
  std::string pre_type = pre.get_string("type");
  if (pre_type == "ByteLevel" && pre.get_bool("use_regex", true)) {
    tok->pre_ = PreTokenizer::Gpt2;
  } else if (pre_type == "Sequence") {
    const Json& list = pre["pretokenizers"];
    if (list.size() != 2 || list.at(0).get_string("type") != "Split" ||
        list.at(1).get_string("type") != "ByteLevel" || list.at(1).get_bool("use_regex", true)) {
      fail("tokenizer: unsupported pre-tokenizer sequence");
    }
    std::string pattern = list.at(0)["pattern"].get_string("Regex");
    if (list.at(0).get_string("behavior") != "Isolated") fail("tokenizer: unsupported split");
    if (pattern == kQwen2Pattern) {
      tok->pre_ = PreTokenizer::Qwen2;
    } else if (pattern == kLlama3Pattern) {
      tok->pre_ = PreTokenizer::Llama3;
    } else if (pattern == kQwen35Pattern) {
      tok->pre_ = PreTokenizer::Qwen35;
    } else {
      fail("tokenizer: unsupported pre-tokenizer pattern");
    }
  } else {
    fail("tokenizer: unsupported pre-tokenizer");
  }

  const Json& vocab = model["vocab"];
  const Json& added = doc["added_tokens"];
  int32_t max_id = -1;
  for (const Json& v : vocab.items())
    max_id = std::max<int32_t>(max_id, static_cast<int32_t>(v.as_int()));
  for (const Json& a : added.items())
    max_id = std::max<int32_t>(max_id, static_cast<int32_t>(a.get_int("id", -1)));
  if (max_id < 0) fail("tokenizer: empty vocabulary");
  tok->tokens_.assign(static_cast<size_t>(max_id) + 1, std::string());
  tok->token_kind_.assign(static_cast<size_t>(max_id) + 1, kUnused);
  tok->vocab_.reserve(vocab.size() * 2);
  std::string raw;
  for (size_t i = 0; i < vocab.size(); ++i) {
    int32_t id = static_cast<int32_t>(vocab.items()[i].as_int());
    if (!unmap_bytes(vocab.keys()[i], raw))
      fail("tokenizer: token is not byte-level: " + vocab.keys()[i]);
    tok->tokens_[static_cast<size_t>(id)] = raw;
    tok->token_kind_[static_cast<size_t>(id)] = kNormal;
    tok->vocab_[raw] = id;
  }
  for (int b = 0; b < 256; ++b) {
    auto it = tok->vocab_.find(std::string(1, static_cast<char>(b)));
    if (it == tok->vocab_.end()) fail("tokenizer: missing byte token");
    tok->byte_ids_[b] = it->second;
  }
  const Json& merges = model["merges"];
  tok->merges_.reserve(merges.size() * 2);
  std::string left;
  std::string right;
  for (size_t rank = 0; rank < merges.size(); ++rank) {
    const Json& m = merges.at(rank);
    std::string a;
    std::string b;
    if (m.is_string()) {
      const std::string& s = m.as_string();
      size_t sp = s.find(' ');
      if (sp == std::string::npos) fail("tokenizer: malformed merge");
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    } else {
      a = m.at(0).as_string();
      b = m.at(1).as_string();
    }
    if (!unmap_bytes(a, left) || !unmap_bytes(b, right)) fail("tokenizer: malformed merge");
    auto ia = tok->vocab_.find(left);
    auto ib = tok->vocab_.find(right);
    auto im = tok->vocab_.find(left + right);
    if (ia == tok->vocab_.end() || ib == tok->vocab_.end() || im == tok->vocab_.end()) {
      fail("tokenizer: merge refers to unknown tokens");
    }
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ia->second)) << 32) |
                   static_cast<uint32_t>(ib->second);
    tok->merges_.emplace(key, Merge{static_cast<int32_t>(rank), im->second});
  }
  for (const Json& a : added.items()) {
    int32_t id = static_cast<int32_t>(a.get_int("id", -1));
    std::string content = a.get_string("content");
    if (id < 0 || content.empty()) continue;
    tok->tokens_[static_cast<size_t>(id)] = content;
    tok->token_kind_[static_cast<size_t>(id)] = a.get_bool("special", false) ? kSpecial : kAdded;
    tok->added_[content] = id;
    tok->max_added_len_ = std::max(tok->max_added_len_, content.size());
    tok->added_first_byte_[static_cast<uint8_t>(content[0])] = true;
  }
  return tok;
}

const std::string& Tokenizer::token_bytes(int32_t id) const {
  static const std::string empty;
  if (id < 0 || id >= size()) return empty;
  return tokens_[static_cast<size_t>(id)];
}

int32_t Tokenizer::find_added(std::string_view content) const {
  auto it = added_.find(std::string(content));
  return it == added_.end() ? -1 : it->second;
}

bool Tokenizer::is_special(int32_t id) const {
  return id >= 0 && id < size() && token_kind_[static_cast<size_t>(id)] == kSpecial;
}

bool Tokenizer::is_added(int32_t id) const {
  if (id < 0 || id >= size()) return false;
  uint8_t k = token_kind_[static_cast<size_t>(id)];
  return k == kAdded || k == kSpecial;
}

std::vector<std::pair<size_t, size_t>> Tokenizer::split(std::string_view text) const {
  DecodedText d = utf8_decode(text);
  switch (pre_) {
    case PreTokenizer::Gpt2:
      return split_gpt2(d);
    case PreTokenizer::Llama3:
      return split_llama3(d, false, 3);
    case PreTokenizer::Qwen2:
      return split_llama3(d, false, 1);
    case PreTokenizer::Qwen35:
      return split_llama3(d, true, 1);
  }
  return {};
}

std::vector<std::string> Tokenizer::pre_tokenize(std::string_view text) const {
  std::string normalized = nfc_ ? nfc(text) : std::string(text);
  std::vector<std::string> out;
  for (auto [b, e] : split(normalized)) out.push_back(normalized.substr(b, e - b));
  return out;
}

void Tokenizer::bpe(std::string_view piece, std::vector<int32_t>& out) const {
  const size_t n = piece.size();
  if (n == 1) {
    out.push_back(byte_ids_[static_cast<uint8_t>(piece[0])]);
    return;
  }
  std::string key(piece);
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      out.insert(out.end(), it->second.begin(), it->second.end());
      return;
    }
  }
  struct Symbol {
    int32_t id;
    int32_t prev;
    int32_t next;
    bool alive;
  };
  struct Candidate {
    int32_t rank;
    int32_t pos;
    int32_t left;
    int32_t right;
    int32_t merged;
    bool operator<(const Candidate& o) const {
      if (rank != o.rank) return rank > o.rank;
      return pos > o.pos;
    }
  };
  std::vector<Symbol> sym(n);
  for (size_t i = 0; i < n; ++i) {
    sym[i] = {byte_ids_[static_cast<uint8_t>(piece[i])], static_cast<int32_t>(i) - 1,
              i + 1 < n ? static_cast<int32_t>(i + 1) : -1, true};
  }
  std::priority_queue<Candidate> heap;
  auto push_pair = [&](int32_t pos) {
    if (pos < 0) return;
    int32_t next = sym[static_cast<size_t>(pos)].next;
    if (next < 0) return;
    int32_t a = sym[static_cast<size_t>(pos)].id;
    int32_t b = sym[static_cast<size_t>(next)].id;
    auto it = merges_.find((static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
                           static_cast<uint32_t>(b));
    if (it != merges_.end()) heap.push({it->second.rank, pos, a, b, it->second.id});
  };
  for (size_t i = 0; i + 1 < n; ++i) push_pair(static_cast<int32_t>(i));
  while (!heap.empty()) {
    Candidate c = heap.top();
    heap.pop();
    Symbol& s = sym[static_cast<size_t>(c.pos)];
    if (!s.alive || s.next < 0 || s.id != c.left) continue;
    Symbol& r = sym[static_cast<size_t>(s.next)];
    if (r.id != c.right) continue;
    s.id = c.merged;
    r.alive = false;
    s.next = r.next;
    if (r.next >= 0) sym[static_cast<size_t>(r.next)].prev = c.pos;
    push_pair(s.prev);
    push_pair(c.pos);
  }
  std::vector<int32_t> ids;
  for (int32_t i = 0; i >= 0; i = sym[static_cast<size_t>(i)].next)
    ids.push_back(sym[static_cast<size_t>(i)].id);
  out.insert(out.end(), ids.begin(), ids.end());
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (cache_.size() > 200000) cache_.clear();
  cache_.emplace(std::move(key), std::move(ids));
}

void Tokenizer::encode_plain(std::string_view text, std::vector<int32_t>& out) const {
  if (text.empty()) return;
  std::string normalized = nfc_ ? nfc(text) : std::string(text);
  for (auto [b, e] : split(normalized)) bpe(std::string_view(normalized).substr(b, e - b), out);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
  std::vector<int32_t> out;
  if (!parse_special || added_.empty()) {
    encode_plain(text, out);
    return out;
  }
  size_t start = 0;
  size_t i = 0;
  while (i < text.size()) {
    bool matched = false;
    if (added_first_byte_[static_cast<uint8_t>(text[i])]) {
      size_t longest = std::min(max_added_len_, text.size() - i);
      for (size_t len = longest; len >= 1; --len) {
        auto it = added_.find(std::string(text.substr(i, len)));
        if (it == added_.end()) continue;
        encode_plain(text.substr(start, i - start), out);
        out.push_back(it->second);
        i += len;
        start = i;
        matched = true;
        break;
      }
    }
    if (!matched) ++i;
  }
  encode_plain(text.substr(start), out);
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  std::string raw;
  for (int32_t id : ids) {
    if (id < 0 || id >= size()) continue;
    if (skip_special && is_special(id)) continue;
    raw += tokens_[static_cast<size_t>(id)];
  }
  return utf8_sanitize(raw);
}

std::string StreamDecoder::push(int32_t id) {
  pending_ += tokenizer_.token_bytes(id);
  size_t k = utf8_complete_prefix(pending_);
  std::string out = utf8_sanitize(std::string_view(pending_).substr(0, k));
  pending_.erase(0, k);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out = utf8_sanitize(pending_);
  pending_.clear();
  return out;
}

std::string render_chat(ChatFormat format, const std::vector<ChatMessage>& messages,
                        bool add_generation_prompt, int thinking) {
  if (messages.empty()) fail("chat: no messages");
  const bool q35 = format == ChatFormat::Qwen35;
  std::vector<std::string> contents;
  for (const ChatMessage& m : messages) contents.push_back(q35 ? py_strip(m.content) : m.content);
  std::string out;
  size_t first = 0;
  if (messages[0].role == "system") {
    out += "<|im_start|>system\n" + contents[0] + "<|im_end|>\n";
    first = 1;
  }
  long last_query = static_cast<long>(messages.size()) - 1;
  bool found = false;
  for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
    if (messages[static_cast<size_t>(i)].role == "user" &&
        !is_tool_response(contents[static_cast<size_t>(i)])) {
      last_query = i;
      found = true;
      break;
    }
  }
  if (q35 && !found) fail("chat: no user query found in messages");
  for (size_t i = first; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    std::string content = contents[i];
    if (m.role == "user" || (m.role == "system" && !q35)) {
      out += "<|im_start|>" + m.role + "\n" + content + "<|im_end|>\n";
    } else if (m.role == "system") {
      fail("chat: system message must be first");
    } else if (m.role == "assistant") {
      std::string reasoning = m.reasoning;
      if (reasoning.empty()) split_reasoning(content, reasoning);
      bool after_query = static_cast<long>(i) > last_query;
      if (q35) {
        reasoning = py_strip(reasoning);
        if (after_query) {
          out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
        } else {
          out += "<|im_start|>assistant\n" + content;
        }
      } else {
        bool last = i + 1 == messages.size();
        if (after_query && (last || !reasoning.empty())) {
          std::string r = lstrip_newlines(rstrip_newlines(reasoning));
          out +=
              "<|im_start|>assistant\n<think>\n" + r + "\n</think>\n\n" + lstrip_newlines(content);
        } else {
          out += "<|im_start|>assistant\n" + content;
        }
      }
      out += "<|im_end|>\n";
    } else {
      fail("chat: unsupported role '" + m.role + "'");
    }
  }
  if (add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    if (q35) {
      out += thinking == 1 ? "<think>\n" : "<think>\n\n</think>\n\n";
    } else if (thinking == 0) {
      out += "<think>\n\n</think>\n\n";
    }
  }
  return out;
}

}
