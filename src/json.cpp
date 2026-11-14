#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "common.h"

namespace infer {

void append_utf8(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : s_(text) {}

  Json parse_document() {
    skip_ws();
    Json v = parse_value(0);
    skip_ws();
    if (pos_ != s_.size()) error("unexpected trailing data");
    return v;
  }

 private:
  [[noreturn]] void error(const char* what) const {
    fail(strf("json parse error at offset %zu: %s", pos_, what));
  }

  void skip_ws() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c != ' ' && c != '\n' && c != '\r' && c != '\t') break;
      ++pos_;
    }
  }

  bool consume(char c) {
    if (pos_ < s_.size() && s_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect_word(std::string_view word) {
    if (s_.substr(pos_, word.size()) != word) error("invalid literal");
    pos_ += word.size();
  }

  Json parse_value(int depth) {
    if (depth > 256) error("nesting too deep");
    if (pos_ >= s_.size()) error("unexpected end of input");
    char c = s_[pos_];
    if (c == '{') return parse_object(depth);
    if (c == '[') return parse_array(depth);
    if (c == '"') {
      Json v;
      v.type_ = Json::Type::String;
      parse_string(v.str_);
      return v;
    }
    if (c == 't') {
      expect_word("true");
      return Json(true);
    }
    if (c == 'f') {
      expect_word("false");
      return Json(false);
    }
    if (c == 'n') {
      expect_word("null");
      return Json();
    }
    return parse_number();
  }

  Json parse_object(int depth) {
    Json v = Json::object();
    ++pos_;
    skip_ws();
    if (consume('}')) return v;
    for (;;) {
      skip_ws();
      if (pos_ >= s_.size() || s_[pos_] != '"') error("expected object key");
      std::string key;
      parse_string(key);
      skip_ws();
      if (!consume(':')) error("expected ':'");
      skip_ws();
      v.keys_.push_back(std::move(key));
      v.values_.push_back(parse_value(depth + 1));
      skip_ws();
      if (consume(',')) continue;
      if (consume('}')) return v;
      error("expected ',' or '}'");
    }
  }

  Json parse_array(int depth) {
    Json v = Json::array();
    ++pos_;
    skip_ws();
    if (consume(']')) return v;
    for (;;) {
      skip_ws();
      v.values_.push_back(parse_value(depth + 1));
      skip_ws();
      if (consume(',')) continue;
      if (consume(']')) return v;
      error("expected ',' or ']'");
    }
  }

  uint32_t parse_hex4() {
    if (pos_ + 4 > s_.size()) error("truncated unicode escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        error("invalid unicode escape");
      }
    }
    return v;
  }

  void parse_string(std::string& out) {
    ++pos_;
    for (;;) {
      size_t start = pos_;
      while (pos_ < s_.size()) {
        char c = s_[pos_];
        if (c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20) break;
        ++pos_;
      }
      out.append(s_.data() + start, pos_ - start);
      if (pos_ >= s_.size()) error("unterminated string");
      char c = s_[pos_++];
      if (c == '"') return;
      if (c != '\\') error("control character in string");
      if (pos_ >= s_.size()) error("unterminated escape");
      char e = s_[pos_++];
      switch (e) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          uint32_t cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos_ + 6 <= s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
              size_t save = pos_;
              pos_ += 2;
              uint32_t lo = parse_hex4();
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                pos_ = save;
                cp = 0xFFFD;
              }
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD;
          }
          append_utf8(cp, out);
          break;
        }
        default:
          error("invalid escape");
      }
    }
  }

  Json parse_number() {
    size_t start = pos_;
    consume('-');
    if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) error("invalid number");
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        ++pos_;
      } else {
        break;
      }
    }
    std::string buf(s_.substr(start, pos_ - start));
    char* end = nullptr;
    double v = std::strtod(buf.c_str(), &end);
    if (end != buf.c_str() + buf.size()) error("invalid number");
    return Json(v);
  }

  std::string_view s_;
  size_t pos_ = 0;
};

Json Json::array() {
  Json v;
  v.type_ = Type::Array;
  return v;
}

Json Json::object() {
  Json v;
  v.type_ = Type::Object;
  return v;
}

Json Json::parse(std::string_view text) { return JsonParser(text).parse_document(); }

bool Json::as_bool() const {
  if (type_ != Type::Bool) fail("json value is not a bool");
  return num_ != 0.0;
}

double Json::as_number() const {
  if (type_ != Type::Number) fail("json value is not a number");
  return num_;
}

int64_t Json::as_int() const {
  double v = as_number();
  if (v != std::floor(v) || std::fabs(v) > 9007199254740992.0)
    fail("json number is not an integer");
  return static_cast<int64_t>(v);
}

const std::string& Json::as_string() const {
  if (type_ != Type::String) fail("json value is not a string");
  return str_;
}

size_t Json::size() const {
  if (type_ == Type::Array || type_ == Type::Object) return values_.size();
  return 0;
}

const Json& Json::at(size_t i) const {
  if (i >= values_.size()) fail("json index out of range");
  return values_[i];
}

const Json* Json::find(std::string_view key) const {
  if (type_ != Type::Object) return nullptr;
  for (size_t i = 0; i < keys_.size(); ++i) {
    if (keys_[i] == key) return &values_[i];
  }
  return nullptr;
}

const Json& Json::operator[](std::string_view key) const {
  static const Json null_value;
  const Json* v = find(key);
  return v != nullptr ? *v : null_value;
}

Json& Json::set(std::string_view key, Json value) {
  if (type_ != Type::Object) fail("json value is not an object");
  for (size_t i = 0; i < keys_.size(); ++i) {
    if (keys_[i] == key) {
      values_[i] = std::move(value);
      return *this;
    }
  }
  keys_.emplace_back(key);
  values_.push_back(std::move(value));
  return *this;
}

Json& Json::push(Json value) {
  if (type_ != Type::Array) fail("json value is not an array");
  values_.push_back(std::move(value));
  return *this;
}

std::string Json::get_string(std::string_view key, const std::string& fallback) const {
  const Json* v = find(key);
  return v != nullptr && v->is_string() ? v->str_ : fallback;
}

double Json::get_number(std::string_view key, double fallback) const {
  const Json* v = find(key);
  return v != nullptr && v->is_number() ? v->num_ : fallback;
}

int64_t Json::get_int(std::string_view key, int64_t fallback) const {
  const Json* v = find(key);
  return v != nullptr && v->is_number() ? v->as_int() : fallback;
}

bool Json::get_bool(std::string_view key, bool fallback) const {
  const Json* v = find(key);
  return v != nullptr && v->is_bool() ? v->num_ != 0.0 : fallback;
}

void json_escape(std::string_view s, std::string& out) {
  out += '"';
  for (char ch : s) {
    unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

void Json::dump_to(std::string& out) const {
  switch (type_) {
    case Type::Null:
      out += "null";
      break;
    case Type::Bool:
      out += num_ != 0.0 ? "true" : "false";
      break;
    case Type::Number: {
      char buf[40];
      if (!std::isfinite(num_)) {
        out += "null";
      } else if (num_ == std::floor(num_) && std::fabs(num_) < 9007199254740992.0) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(num_));
        out += buf;
      } else {
        std::snprintf(buf, sizeof(buf), "%.9g", num_);
        out += buf;
      }
      break;
    }
    case Type::String:
      json_escape(str_, out);
      break;
    case Type::Array:
      out += '[';
      for (size_t i = 0; i < values_.size(); ++i) {
        if (i) out += ',';
        values_[i].dump_to(out);
      }
      out += ']';
      break;
    case Type::Object:
      out += '{';
      for (size_t i = 0; i < values_.size(); ++i) {
        if (i) out += ',';
        json_escape(keys_[i], out);
        out += ':';
        values_[i].dump_to(out);
      }
      out += '}';
      break;
  }
}

std::string Json::dump() const {
  std::string out;
  dump_to(out);
  return out;
}

}
