#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infer {

class Json {
 public:
  enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool v) : type_(Type::Bool), num_(v ? 1.0 : 0.0) {}
  Json(int v) : type_(Type::Number), num_(v) {}
  Json(int64_t v) : type_(Type::Number), num_(static_cast<double>(v)) {}
  Json(uint64_t v) : type_(Type::Number), num_(static_cast<double>(v)) {}
  Json(double v) : type_(Type::Number), num_(v) {}
  Json(float v) : type_(Type::Number), num_(v) {}
  Json(const char* s) : type_(Type::String), str_(s) {}
  Json(std::string s) : type_(Type::String), str_(std::move(s)) {}
  Json(std::string_view s) : type_(Type::String), str_(s) {}

  static Json array();
  static Json object();
  static Json parse(std::string_view text);

  std::string dump() const;
  void dump_to(std::string& out) const;

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const;
  double as_number() const;
  int64_t as_int() const;
  const std::string& as_string() const;

  size_t size() const;
  const Json& at(size_t i) const;
  const std::vector<Json>& items() const { return values_; }
  const std::vector<std::string>& keys() const { return keys_; }

  const Json* find(std::string_view key) const;
  const Json& operator[](std::string_view key) const;
  bool contains(std::string_view key) const { return find(key) != nullptr; }

  Json& set(std::string_view key, Json value);
  Json& push(Json value);

  std::string get_string(std::string_view key, const std::string& fallback = {}) const;
  double get_number(std::string_view key, double fallback) const;
  int64_t get_int(std::string_view key, int64_t fallback) const;
  bool get_bool(std::string_view key, bool fallback) const;

 private:
  Type type_ = Type::Null;
  double num_ = 0.0;
  std::string str_;
  std::vector<std::string> keys_;
  std::vector<Json> values_;

  friend class JsonParser;
};

void json_escape(std::string_view s, std::string& out);
void append_utf8(uint32_t cp, std::string& out);

}
