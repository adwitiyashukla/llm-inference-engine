import shutil
import subprocess
import sys

import unicodedata2 as ud
from tokenizers import normalizers

NFC = normalizers.NFC()
NFD = normalizers.NFD()


def category_ranges(prefix):
    out = []
    start = None
    for cp in range(0x110000):
        hit = ud.category(chr(cp))[0] == prefix
        if hit and start is None:
            start = cp
        if not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def skip(cp):
    return 0xD800 <= cp <= 0xDFFF


def known_mark(cp):
    probe = "a" + chr(cp) + "\u0334" if ud.combining(chr(cp)) > 1 else "a\u0301" + chr(cp)
    return NFD.normalize_str(probe) == ud.normalize("NFD", probe)


def combining_ranges():
    classes = {}
    for cp in range(0x110000):
        if skip(cp):
            continue
        ccc = ud.combining(chr(cp))
        if ccc and known_mark(cp):
            classes[cp] = ccc
    out = []
    for cp in sorted(classes):
        if out and out[-1][1] == cp - 1 and out[-1][2] == classes[cp]:
            out[-1] = (out[-1][0], cp, classes[cp])
        else:
            out.append((cp, cp, classes[cp]))
    return out


def decompositions():
    decomp = []
    compose = []
    for cp in range(0x110000):
        if skip(cp) or 0xAC00 <= cp <= 0xD7A3:
            continue
        d = ud.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        if NFD.normalize_str(chr(cp)) != ud.normalize("NFD", chr(cp)):
            continue
        parts = [int(x, 16) for x in d.split()]
        decomp.append((cp, parts[0], parts[1] if len(parts) > 1 else 0))
        if len(parts) == 2 and NFC.normalize_str(chr(parts[0]) + chr(parts[1])) == chr(cp):
            compose.append((parts[0], parts[1], cp))
    compose.sort()
    return decomp, compose


def emit(name, rows, out):
    flat = [v for row in rows for v in row]
    out.append(f"const uint32_t {name}[] = {{")
    line = "   "
    for v in flat:
        item = f" 0x{v:x},"
        if len(line) + len(item) > 100:
            out.append(line)
            line = "   "
        line += item
    out.append(line)
    out.append("};")
    out.append(f"const size_t {name}Size = {len(rows)};")
    out.append("")


def main():
    if ud.unidata_version != "16.0.0":
        sys.exit("needs unicodedata2==16.0.0, the Unicode version of the tokenizers regex engine")
    path = sys.argv[1]
    decomp, compose = decompositions()
    out = ['#include "unicode.h"', "", "namespace infer {", ""]
    emit("kLetterRanges", category_ranges("L"), out)
    emit("kMarkRanges", category_ranges("M"), out)
    emit("kNumberRanges", category_ranges("N"), out)
    emit("kCombiningClasses", combining_ranges(), out)
    emit("kDecompositions", decomp, out)
    emit("kCompositions", compose, out)
    out.append(f'const char* const kUnicodeVersion = "{ud.unidata_version}";')
    out.append("")
    out.append("}  // namespace infer")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    if shutil.which("clang-format"):
        subprocess.run(["clang-format", "-i", path], check=True)


if __name__ == "__main__":
    main()
