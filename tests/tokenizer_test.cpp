#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "test_models.h"
#include "tokenizer.h"
#include "unicode.h"

using namespace infer;

namespace {

std::unique_ptr<Tokenizer> tiny() { return Tokenizer::from_json(tiny_tokenizer_json()); }

std::vector<std::string> pieces(const Tokenizer& t, const std::string& text) {
  return t.pre_tokenize(text);
}

}

TEST(utf8_decoding_and_repair) {
  DecodedText d = utf8_decode("a\xC3\xA9\xE2\x82");
  EXPECT(d.cps.size() == 3);
  EXPECT(d.cps[0] == 'a');
  EXPECT(d.cps[1] == 0xE9);
  EXPECT(d.cps[2] == 0xFFFD);
  EXPECT(d.offsets.size() == 4);
  EXPECT(d.offsets[3] == 5);
  EXPECT(utf8_sanitize("ok\xFF") == "ok\xEF\xBF\xBD");
  EXPECT(utf8_sanitize("\xED\xA0\x80") == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD");
  EXPECT(utf8_sanitize("\xEF\xBF\xBD") == "\xEF\xBF\xBD");
  EXPECT(utf8_complete_prefix("ab\xE4\xB8") == 2);
  EXPECT(utf8_complete_prefix("ab\xE4\xB8\xAD") == 5);
  EXPECT(utf8_complete_prefix("ab\xFF") == 3);
  EXPECT(utf8_complete_prefix("ab\xF0\x9F\x98") == 2);
  EXPECT(utf8_complete_prefix("") == 0);
}

TEST(nfc_normalization) {
  EXPECT(nfc("e\xCC\x81") == "\xC3\xA9");
  EXPECT(nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8") == "\xEA\xB0\x81");
  EXPECT(nfc("\xE2\x84\xA6") == "\xCE\xA9");
  EXPECT(nfc("a\xCC\x81\xCC\xA3") == "\xE1\xBA\xA1\xCC\x81");
  EXPECT(nfc("plain ascii text") == "plain ascii text");
  EXPECT(nfc("\xE0\xA5\x98") == "\xE0\xA4\x95\xE0\xA4\xBC");
}

TEST(character_classes) {
  EXPECT(is_letter('a'));
  EXPECT(is_letter(0x4E2D));
  EXPECT(!is_letter('7'));
  EXPECT(is_number('7'));
  EXPECT(is_number(0x2163));
  EXPECT(is_mark(0x0301));
  EXPECT(!is_mark('a'));
  EXPECT(is_whitespace(0x3000));
  EXPECT(!is_whitespace(0x200B));
  EXPECT(combining_class(0x0301) == 230);
  EXPECT(combining_class('a') == 0);
}

TEST(qwen_pre_tokenizer_splits) {
  auto t = tiny();
  EXPECT(pieces(*t, "Hello world") == (std::vector<std::string>{"Hello", " world"}));
  EXPECT(pieces(*t, "  hi") == (std::vector<std::string>{" ", " hi"}));
  EXPECT(pieces(*t, "a\n\nb") == (std::vector<std::string>{"a", "\n\n", "b"}));
  EXPECT(pieces(*t, "x's 'LL") == (std::vector<std::string>{"x", "'s", " '", "LL"}));
  EXPECT(pieces(*t, "123") == (std::vector<std::string>{"1", "2", "3"}));
  EXPECT(pieces(*t, "hi!!!\n") == (std::vector<std::string>{"hi", "!!!\n"}));
  EXPECT(pieces(*t, "end  ") == (std::vector<std::string>{"end", "  "}));
  EXPECT(pieces(*t, "tab\t\tx") == (std::vector<std::string>{"tab", "\t", "\tx"}));
}

TEST(bpe_applies_merges_by_rank) {
  auto t = tiny();
  EXPECT(t->encode(" the") == (std::vector<int32_t>{261}));
  EXPECT(t->encode("the") == (std::vector<int32_t>{257}));
  EXPECT(t->encode("and") == (std::vector<int32_t>{259}));
  EXPECT(t->encode("in") == (std::vector<int32_t>{260}));
  EXPECT(t->encode("xyz").size() == 3);
  EXPECT(t->encode("").empty());
}

TEST(special_tokens_and_decoding) {
  auto t = tiny();
  const int32_t start = t->find_added("<|im_start|>");
  const int32_t end = t->find_added("<|im_end|>");
  EXPECT(start >= 0 && end >= 0);
  EXPECT(t->is_special(start));
  EXPECT(!t->is_special(257));
  std::vector<int32_t> ids = t->encode("<|im_start|>the<|im_end|>");
  EXPECT(ids == (std::vector<int32_t>{start, 257, end}));
  EXPECT(t->encode("<|im_start|>", false).size() > 1);
  EXPECT(t->decode(ids) == "<|im_start|>the<|im_end|>");
  EXPECT(t->decode(ids, true) == "the");
  const std::string text = "caf\xC3\xA9 \xE4\xB8\xAD and the end";
  EXPECT(t->decode(t->encode(text)) == text);
}

TEST(stream_decoder_waits_for_whole_characters) {
  auto t = tiny();
  std::vector<int32_t> ids = t->encode("\xE4\xB8\xAD!");
  EXPECT(ids.size() == 4);
  if (ids.size() != 4) return;
  StreamDecoder d(*t);
  EXPECT(d.push(ids[0]).empty());
  EXPECT(d.push(ids[1]).empty());
  EXPECT(d.push(ids[2]) == "\xE4\xB8\xAD");
  EXPECT(d.push(ids[3]) == "!");
  EXPECT(d.flush().empty());
  StreamDecoder cut(*t);
  EXPECT(cut.push(ids[0]).empty());
  EXPECT(cut.flush() == "\xEF\xBF\xBD");
}

TEST(chat_templates) {
  std::vector<ChatMessage> sys = {{"system", " be brief ", {}}, {"user", "hi", {}}};
  EXPECT(render_chat(ChatFormat::Qwen35, sys, true, -1) ==
         "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  std::vector<ChatMessage> one = {{"user", "hi", {}}};
  EXPECT(render_chat(ChatFormat::Qwen35, one, true, 1) ==
         "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
  EXPECT(render_chat(ChatFormat::Qwen3, one, true, -1) ==
         "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
  EXPECT(render_chat(ChatFormat::Qwen3, one, true, 0) ==
         "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  std::vector<ChatMessage> turns = {
      {"user", "q1", {}}, {"assistant", "<think>\nr1\n</think>\n\na1", {}}, {"user", "q2", {}}};
  EXPECT(render_chat(ChatFormat::Qwen35, turns, true, -1) ==
         "<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\na1<|im_end|>\n"
         "<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

int main() { return check::run_all(); }
