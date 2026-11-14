#include "unicode.h"

#include <algorithm>
#include <unordered_map>

#include "json.h"

namespace infer {
namespace {

constexpr uint8_t kFlagLetter = 1;
constexpr uint8_t kFlagMark = 2;
constexpr uint8_t kFlagNumber = 4;

constexpr uint32_t kSBase = 0xAC00;
constexpr uint32_t kLBase = 0x1100;
constexpr uint32_t kVBase = 0x1161;
constexpr uint32_t kTBase = 0x11A7;
constexpr uint32_t kLCount = 19;
constexpr uint32_t kVCount = 21;
constexpr uint32_t kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount;
constexpr uint32_t kSCount = kLCount * kNCount;

bool in_ranges(const uint32_t* r, size_t n, uint32_t cp) {
  size_t lo = 0;
  size_t hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (r[2 * mid + 1] < cp) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < n && r[2 * lo] <= cp;
}

uint8_t ccc_lookup(uint32_t cp) {
  size_t lo = 0;
  size_t hi = kCombiningClassesSize;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (kCombiningClasses[3 * mid + 1] < cp) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < kCombiningClassesSize && kCombiningClasses[3 * lo] <= cp) {
    return static_cast<uint8_t>(kCombiningClasses[3 * lo + 2]);
  }
  return 0;
}

struct BmpTables {
  std::vector<uint8_t> flags;
  std::vector<uint8_t> ccc;
  BmpTables() : flags(0x10000), ccc(0x10000) {
    for (uint32_t cp = 0; cp < 0x10000; ++cp) {
      uint8_t f = 0;
      if (in_ranges(kLetterRanges, kLetterRangesSize, cp)) f |= kFlagLetter;
      if (in_ranges(kMarkRanges, kMarkRangesSize, cp)) f |= kFlagMark;
      if (in_ranges(kNumberRanges, kNumberRangesSize, cp)) f |= kFlagNumber;
      flags[cp] = f;
      ccc[cp] = ccc_lookup(cp);
    }
  }
};

const BmpTables& bmp() {
  static const BmpTables tables;
  return tables;
}

struct CompositionMap {
  std::unordered_map<uint64_t, uint32_t> map;
  CompositionMap() {
    map.reserve(kCompositionsSize * 2);
    for (size_t i = 0; i < kCompositionsSize; ++i) {
      uint64_t key = (static_cast<uint64_t>(kCompositions[3 * i]) << 32) | kCompositions[3 * i + 1];
      map.emplace(key, kCompositions[3 * i + 2]);
    }
  }
};

const CompositionMap& compositions() {
  static const CompositionMap m;
  return m;
}

const uint32_t* find_decomposition(uint32_t cp) {
  size_t lo = 0;
  size_t hi = kDecompositionsSize;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (kDecompositions[3 * mid] < cp) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < kDecompositionsSize && kDecompositions[3 * lo] == cp) return kDecompositions + 3 * lo;
  return nullptr;
}

void decompose(uint32_t cp, std::vector<uint32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    uint32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    uint32_t t = s % kTCount;
    if (t != 0) out.push_back(kTBase + t);
    return;
  }
  const uint32_t* d = find_decomposition(cp);
  if (d == nullptr) {
    out.push_back(cp);
    return;
  }
  decompose(d[1], out);
  if (d[2] != 0) decompose(d[2], out);
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  }
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase &&
      b < kTBase + kTCount) {
    return a + (b - kTBase);
  }
  const auto& m = compositions().map;
  auto it = m.find((static_cast<uint64_t>(a) << 32) | b);
  return it == m.end() ? 0 : it->second;
}

inline bool is_cont(unsigned char c) { return (c & 0xC0) == 0x80; }

size_t decode_one(std::string_view s, size_t i, uint32_t& cp) {
  const size_t n = s.size();
  unsigned char b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    cp = b0;
    return 1;
  }
  size_t need = 0;
  unsigned char lo = 0x80;
  unsigned char hi = 0xBF;
  uint32_t value = 0;
  if (b0 >= 0xC2 && b0 <= 0xDF) {
    need = 1;
    value = b0 & 0x1F;
  } else if (b0 >= 0xE0 && b0 <= 0xEF) {
    need = 2;
    value = b0 & 0x0F;
    if (b0 == 0xE0) lo = 0xA0;
    if (b0 == 0xED) hi = 0x9F;
  } else if (b0 >= 0xF0 && b0 <= 0xF4) {
    need = 3;
    value = b0 & 0x07;
    if (b0 == 0xF0) lo = 0x90;
    if (b0 == 0xF4) hi = 0x8F;
  } else {
    cp = 0xFFFD;
    return 1;
  }
  size_t used = 1;
  for (size_t k = 0; k < need; ++k) {
    if (i + used >= n) {
      cp = 0xFFFD;
      return used;
    }
    unsigned char c = static_cast<unsigned char>(s[i + used]);
    bool ok = k == 0 ? (c >= lo && c <= hi) : is_cont(c);
    if (!ok) {
      cp = 0xFFFD;
      return used;
    }
    value = (value << 6) | (c & 0x3F);
    ++used;
  }
  cp = value;
  return used;
}

}

bool is_letter(uint32_t cp) {
  if (cp < 0x10000) return (bmp().flags[cp] & kFlagLetter) != 0;
  return in_ranges(kLetterRanges, kLetterRangesSize, cp);
}

bool is_mark(uint32_t cp) {
  if (cp < 0x10000) return (bmp().flags[cp] & kFlagMark) != 0;
  return in_ranges(kMarkRanges, kMarkRangesSize, cp);
}

bool is_number(uint32_t cp) {
  if (cp < 0x10000) return (bmp().flags[cp] & kFlagNumber) != 0;
  return in_ranges(kNumberRanges, kNumberRangesSize, cp);
}

bool is_whitespace(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0d) || cp == 0x20 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f ||
         cp == 0x205f || cp == 0x3000;
}

bool is_python_space(uint32_t cp) { return is_whitespace(cp) || (cp >= 0x1c && cp <= 0x1f); }

uint8_t combining_class(uint32_t cp) {
  if (cp < 0x10000) return bmp().ccc[cp];
  return ccc_lookup(cp);
}

DecodedText utf8_decode(std::string_view s) {
  DecodedText out;
  out.cps.reserve(s.size());
  out.offsets.reserve(s.size() + 1);
  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp = 0;
    size_t len = decode_one(s, i, cp);
    out.cps.push_back(cp);
    out.offsets.push_back(static_cast<uint32_t>(i));
    i += len;
  }
  out.offsets.push_back(static_cast<uint32_t>(s.size()));
  return out;
}

size_t utf8_complete_prefix(std::string_view s) {
  size_t n = s.size();
  size_t start = n;
  for (size_t back = 1; back <= 3 && back <= n; ++back) {
    unsigned char c = static_cast<unsigned char>(s[n - back]);
    if (!is_cont(c)) {
      start = n - back;
      break;
    }
  }
  if (start == n) return n;
  uint32_t cp = 0;
  size_t len = decode_one(s, start, cp);
  if (start + len == n && cp == 0xFFFD) {
    unsigned char b0 = static_cast<unsigned char>(s[start]);
    size_t expect = (b0 >= 0xF0 && b0 <= 0xF4)   ? 4
                    : (b0 >= 0xE0 && b0 <= 0xEF) ? 3
                    : (b0 >= 0xC2 && b0 <= 0xDF) ? 2
                                                 : 1;
    if (expect > n - start) return start;
  }
  return n;
}

std::string utf8_sanitize(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp = 0;
    size_t len = decode_one(s, i, cp);
    if (cp == 0xFFFD && !(len == 3 && s.substr(i, 3) == "\xEF\xBF\xBD")) {
      out += "\xEF\xBF\xBD";
    } else {
      out.append(s.data() + i, len);
    }
    i += len;
  }
  return out;
}

std::string nfc(std::string_view s) {
  bool ascii = true;
  for (char c : s) {
    if (static_cast<unsigned char>(c) >= 0x80) {
      ascii = false;
      break;
    }
  }
  if (ascii) return std::string(s);
  DecodedText d = utf8_decode(s);
  std::vector<uint32_t> buf;
  buf.reserve(d.cps.size() + 8);
  for (uint32_t cp : d.cps) decompose(cp, buf);
  for (size_t i = 1; i < buf.size(); ++i) {
    uint8_t c = combining_class(buf[i]);
    if (c == 0) continue;
    size_t j = i;
    while (j > 0 && combining_class(buf[j - 1]) > c) {
      std::swap(buf[j - 1], buf[j]);
      --j;
    }
  }
  std::vector<uint32_t> out;
  out.reserve(buf.size());
  if (!buf.empty()) {
    size_t starter = 0;
    out.push_back(buf[0]);
    int last_class = combining_class(buf[0]) == 0 ? 0 : 256;
    for (size_t i = 1; i < buf.size(); ++i) {
      uint32_t ch = buf[i];
      int cls = combining_class(ch);
      uint32_t comp = compose_pair(out[starter], ch);
      if (comp != 0 && (last_class < cls || last_class == 0)) {
        out[starter] = comp;
        continue;
      }
      if (cls == 0) starter = out.size();
      last_class = cls;
      out.push_back(ch);
    }
  }
  std::string result;
  result.reserve(s.size());
  for (uint32_t cp : out) append_utf8(cp, result);
  return result;
}

std::string py_strip(std::string_view s) {
  DecodedText d = utf8_decode(s);
  size_t b = 0;
  size_t e = d.cps.size();
  while (b < e && is_python_space(d.cps[b])) ++b;
  while (e > b && is_python_space(d.cps[e - 1])) --e;
  return std::string(s.substr(d.offsets[b], d.offsets[e] - d.offsets[b]));
}

}
