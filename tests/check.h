#pragma once

#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace check {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& cases() {
  static std::vector<Case> all;
  return all;
}

inline int& failures() {
  static int count = 0;
  return count;
}

struct Register {
  Register(const char* name, void (*fn)()) { cases().push_back({name, fn}); }
};

inline int run_all() {
  int failed_cases = 0;
  for (const Case& c : cases()) {
    int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      std::printf("  exception: %s\n", e.what());
      ++failures();
    }
    bool ok = failures() == before;
    if (!ok) ++failed_cases;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
  }
  std::printf("%zu cases, %d failed\n", cases().size(), failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

}

#define TEST(name)                                      \
  static void name();                                   \
  static const check::Register name##_reg(#name, name); \
  static void name()

#define EXPECT(cond)                                                    \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("  %s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
      ++check::failures();                                              \
    }                                                                   \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                                    \
  do {                                                                                            \
    double va_ = static_cast<double>(a);                                                          \
    double vb_ = static_cast<double>(b);                                                          \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                                       \
      std::printf("  %s:%d: %s = %.9g, %s = %.9g, tolerance %.3g\n", __FILE__, __LINE__, #a, va_, \
                  #b, vb_, static_cast<double>(tol));                                             \
      ++check::failures();                                                                        \
    }                                                                                             \
  } while (0)
