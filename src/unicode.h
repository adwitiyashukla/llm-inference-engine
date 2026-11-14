#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace infer {

extern const uint32_t kLetterRanges[];
extern const size_t kLetterRangesSize;
extern const uint32_t kMarkRanges[];
extern const size_t kMarkRangesSize;
extern const uint32_t kNumberRanges[];
extern const size_t kNumberRangesSize;
extern const uint32_t kCombiningClasses[];
extern const size_t kCombiningClassesSize;
extern const uint32_t kDecompositions[];
extern const size_t kDecompositionsSize;
extern const uint32_t kCompositions[];
extern const size_t kCompositionsSize;
extern const char* const kUnicodeVersion;

bool is_letter(uint32_t cp);
bool is_mark(uint32_t cp);
bool is_number(uint32_t cp);
bool is_whitespace(uint32_t cp);
bool is_python_space(uint32_t cp);
uint8_t combining_class(uint32_t cp);

struct DecodedText {
  std::vector<uint32_t> cps;
  std::vector<uint32_t> offsets;
};

DecodedText utf8_decode(std::string_view s);
size_t utf8_complete_prefix(std::string_view s);
std::string utf8_sanitize(std::string_view s);
std::string nfc(std::string_view s);
std::string py_strip(std::string_view s);

}
