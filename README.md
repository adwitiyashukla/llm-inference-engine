# llm-inference-engine

[![ci](https://github.com/adwitiyashukla/llm-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/llm-inference-engine/actions/workflows/ci.yml)

A CPU inference engine for Qwen3.5 and Qwen3 language models, written from scratch in C++20 with no
third party libraries. It reads Hugging Face checkpoints directly, converts them to int8 or int4,
and runs them on hand written AVX2, AVX-VNNI and AVX-512 kernels. On top of the model sits a
continuous batching scheduler, speculative decoding that uses the model's own multi-token prediction
head, and an OpenAI compatible HTTP server.

Qwen3.5 is the reason this project exists. It is a hybrid: 18 of the 24 layers in Qwen3.5-0.8B are
Gated DeltaNet, a linear attention layer that carries a fixed size recurrent state instead of a KV
cache, and the other 6 are full attention with an output gate. The DeltaNet recurrence is
implemented directly, including the state snapshots that let speculative decoding roll back rejected
tokens.

On my 15 W laptop it runs Qwen3.5-0.8B in 8 bits at about 560 tokens per second of prefill and 47 of
decode, which is 2.9 and 1.3 times as fast as llama.cpp on the same model and format. With the
model's own draft head turned on, decode goes up to 74 to 93 tokens per second.

Everything below was measured. Speed numbers come from that laptop, an Intel Core 5 120U (2
performance cores and 8 efficiency cores, 12 threads) with 8 GB of memory, Windows 11, MinGW gcc
15.2. The correctness checks against transformers ran on a 2 core Linux VM, because a float32 copy
of the model in PyTorch does not leave much room on an 8 GB laptop.

## What is in it

- Model code for Qwen3.5 (Gated DeltaNet, gated attention, partial rotary embeddings, zero centered
  RMSNorm, the multi-token prediction head) and Qwen3 (grouped query attention with QK norm).
- A safetensors reader that memory maps the checkpoint, so bf16 runs straight from the download.
- A converter to q8 (int8 with one f16 scale per 32 weights) and q4 (4 bit with a per block scale
  search, with the projections that write into the residual stream kept at 8 bits).
- A byte level BPE tokenizer, the Unicode tables it needs, NFC normalization and both chat
  templates.
- Four kernel sets (scalar, AVX2, AVX-VNNI, AVX-512 VNNI) chosen at startup from CPUID.
- A paged KV cache in 16 token blocks, continuous batching and chunked prefill.
- Speculative decoding with MTP drafts or n-gram prompt lookup. Greedy output is token for token
  identical to plain decoding, and sampling goes through rejection sampling so the output
  distribution does not change.
- An OpenAI compatible server: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`,
  streaming over server sent events, `reasoning_content` for thinking output.

## Correctness

### Tokenizer

`tools/check.py tokenizer` encodes 1,132,064 strings with this tokenizer and with Hugging Face
tokenizers and compares the ids. The strings are every Unicode code point dropped into a template
that hits each pre-tokenizer rule (letters, digits, contractions, punctuation runs, whitespace
before a newline), plus 20,000 random mixes of scripts, emoji, combining marks, numbers and special
tokens. For both the Qwen3.5 and the Qwen3 tokenizer all 1,132,064 match, about 25 million tokens
each, and NFC normalization matches on every string as well.

The chat templates get the same treatment: 2,000 random conversations per template, with system
prompts, multi turn histories, thinking blocks left inside earlier answers, `reasoning_content`
fields and tool responses, rendered against `apply_chat_template`. All 2,000 are byte for byte
identical for each template, including the 335 conversations that the Qwen3.5 template refuses (both
sides refuse them).

### Logits

256 positions of English text through the real checkpoints, against transformers in float32:

| Model | KV cache | Max abs diff | Mean abs diff | Top-1 agreement | Mean KL |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | f32 | 4.3e-5 | 2.8e-6 | 100% | 4.0e-12 |
| Qwen3.5-0.8B | f16 | 6.9e-3 | 3.4e-4 | 100% | 2.0e-8 |
| Qwen3-0.6B | f32 | 1.0e-4 | 4.4e-6 | 100% | 7.0e-12 |
| Qwen3-0.6B | f16 | 6.3e-2 | 1.5e-3 | 99.6% | 4.7e-7 |

f16 is the default KV cache. It costs more on Qwen3 because all 28 of its layers keep a KV cache,
against 6 of the 24 in Qwen3.5.

Four chat prompts decoded greedily for 64 tokens each, against a greedy loop in transformers, give
256 of 256 identical tokens for both models. They are still 256 of 256 with MTP drafting on Qwen3.5
and with n-gram drafting on both.

### Perplexity

WikiText-2 test set, windows of 512 tokens, the first 128 windows (65,408 scored tokens), default
f16 KV cache:

| Model | transformers f32 | bf16 | q8 | q4 |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 23.4121 | 23.4121 | 23.4592 | 24.8853 |
| Qwen3-0.6B | 28.2409 | 28.2409 | 28.2274 | 30.2456 |
| File size, Qwen3.5-0.8B | | 1.75 GB | 836 MB | 649 MB |
| File size, Qwen3-0.6B | | 1.50 GB | 645 MB | 498 MB |

The bf16 runs match transformers to all four printed decimals. q8 is within noise of bf16 (0.2
percent on Qwen3.5, and on Qwen3 it happens to land slightly below), and q4 costs 6 to 7 percent.

q4 needed more work. The first version put every matrix at 4 bits and cost 18 percent on Qwen3.5. To
find out where the damage came from, I converted variants that kept one group of tensors at 8 bits
and scored them on the first 16 windows:

| q4 variant (Qwen3.5-0.8B) | File size | Perplexity, 16 windows |
| --- | --- | --- |
| everything at 4 bits | 578 MB | 32.58 |
| attention projections at 8 bits | 603 MB | 31.73 |
| MLP down projection at 8 bits | 623 MB | 31.04 |
| DeltaNet projections at 8 bits | 672 MB | 30.25 |
| whole MLP at 8 bits | 715 MB | 29.67 |
| projections that write into the residual stream at 8 bits | 649 MB | 28.94 |
| q8 for reference | 836 MB | 27.42 |

The winner is cheap and easy to state: the three projections that add their output back into the
residual stream (attention `o_proj`, DeltaNet `out_proj` and the MLP `down_proj`) stay at 8 bits,
everything else drops to 4. It costs about 70 MB and recovers about 70 percent of the loss. I also
tried weighting the q4 scale search by activation statistics collected on the WikiText-2 training
split. On top of the mixed layout it gained 0.3 percent, which was not worth a calibration step, so
it is not in the code.

## Speed

All numbers in this section come from the laptop (HP OmniBook 5, 8 GB of LPDDR5-5200), where
`infer membw` measures 43.6 GB/s of read bandwidth. Every run is a fresh process. A 15 W chip slows
down once it heats up, so each run starts after a 30 second pause on an idle machine, and the
numbers are medians of three runs.

512 token prompt, 128 generated tokens, 12 threads:

| Model | Weights | Prefill tok/s | Decode tok/s | Weights read per token | Bandwidth at decode |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | q8 | 562 | 47.4 | 799 MB | 37.9 GB/s |
| Qwen3.5-0.8B | q4 | 550 | 57.1 | 620 MB | 35.4 GB/s |
| Qwen3.5-0.8B | bf16 | 223 | 27.4 | 1.5 GB | 41.2 GB/s |
| Qwen3-0.6B | q8 | 516 | 54.1 | 633 MB | 34.3 GB/s |
| Qwen3-0.6B | q4 | 516 | 64.8 | 486 MB | 31.5 GB/s |
| Qwen3-0.6B | bf16 | 235 | 32.9 | 1.2 GB | 39.2 GB/s |

Decode reads every weight once per token, so it is bound by memory bandwidth, and bf16 decode on
Qwen3.5 pulls 41 of the 43.6 GB/s the laptop has. q4 reads 22 percent fewer bytes per token on
Qwen3.5, because the output head and the residual projections stay at 8 bits, and decode gets 20
percent faster. Prefill pushes 256 tokens through each pass over the weights, so it is bound by the
int8 math instead, and q4 prefill ends up about as fast as q8.

### Against llama.cpp and PyTorch

Same laptop, same Qwen3.5-0.8B, both engines at 8 bits with blocks of 32 weights and one f16 scale
(ggml-org's Q8_0 GGUF for llama.cpp). llama-bench ran `-p 512 -n 128 -d 0,512`, so prefill starts
from an empty context and decode runs after 512 tokens of context, the same as `infer bench`. The
runs were interleaved, three of each:

| Engine | Threads | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- |
| this engine, q8 | 12 | 568 | 47.9 |
| llama.cpp b12381, Q8_0 | 12 | 198 | 36.3 |
| llama.cpp b12381, Q8_0 | 10 | 172 | 35.1 |

llama.cpp runs hundreds of models on every kind of hardware, and this engine only has to run two
models well on x86 CPUs, so the comparison says more about how much a narrow engine can get out of
one model than about llama.cpp. For reference, transformers 5.23 on PyTorch 2.13 in bf16 prefills
37.8 and decodes 8.5 tokens per second on Qwen3-0.6B (this engine in bf16: 235 and 32.9). On
Qwen3.5-0.8B it gets 25.5 and 0.7, because without its CUDA kernels transformers runs the DeltaNet
layers in plain PyTorch.

### Speculative decoding

Qwen3.5-0.8B q8, greedy, 256 new tokens. One prompt is open ended (explain a hash map, then write a
word count in Python). The other one is mostly copying (rewrite a paragraph in British spelling).
Accepted is the share of drafted tokens the model kept.

| Drafting | Explain tok/s | Speedup | Accepted | Rewrite tok/s | Speedup | Accepted |
| --- | --- | --- | --- | --- | --- | --- |
| none | 45.8 | 1.00x | - | 47.2 | 1.00x | - |
| MTP, 1 draft | 63.9 | 1.40x | 70% | 67.1 | 1.42x | 92% |
| MTP, 2 drafts (default) | 74.4 | 1.62x | 57% | 92.8 | 1.97x | 89% |
| MTP, 3 drafts | 73.1 | 1.60x | 44% | 102.9 | 2.18x | 81% |
| MTP, 2 drafts, full output head | 56.0 | 1.22x | 60% | 68.2 | 1.44x | 100% |
| n-gram lookup | 45.4 | 0.99x | 10% | 160.0 | 3.39x | 100% |

One pass that checks the pending token plus 2 drafts takes less than 20 percent longer than a plain
decode step, because the weights are read once for all of them. Scoring drafts against only the
first 32,768 token ids loses a few accepted tokens but makes every draft much cheaper, and it comes
out 33 to 36 percent faster than drafting with the full output head. N-gram lookup copies whatever
followed the last matching 2 or 3 tokens earlier in the text. It does nothing for open ended
answers, but on the rewrite prompt it makes decoding 3.4 times faster.

### Threads and batching

Qwen3.5-0.8B q8, two runs each:

| Threads | Prefill tok/s | Decode tok/s |
| --- | --- | --- |
| 1 | 171 | 14.9 |
| 2 | 349 | 36.0 |
| 4 | 341 | 42.5 |
| 6 | 439 | 42.3 |
| 8 | 490 | 43.9 |
| 10 | 538 | 45.4 |
| 12 | 564 | 47.4 |

Decode already reaches 36 tokens per second with 2 threads, since it mostly waits on memory, and the
other cores add about a third on top. Prefill is limited by compute and grows to 564 tokens per
second with all 12 threads.

Several requests at once, 128 token prompts and 128 new tokens each:

| Requests in flight | Decode tok/s per request | Total tok/s |
| --- | --- | --- |
| 1 | 48.5 | 48.5 |
| 2 | 41.5 | 82.8 |
| 4 | 33.0 | 139.9 |
| 8 | 22.5 | 199.2 |

With continuous batching every pass over the weights serves all the requests that are decoding, so 8
requests at once get 4.1 times the total throughput of one.

### Where the time goes

`infer bench --profile` splits a run by phase (q8, 12 threads):

| Phase | Prefill, ms per token | Decode, ms per token |
| --- | --- | --- |
| weight projections | 1.26 (71%) | 12.63 (58%) |
| output head | 0.01 (1%) | 5.56 (25%) |
| DeltaNet | 0.31 (18%) | 1.69 (8%) |
| attention | 0.13 (7%) | 0.83 (4%) |
| everything else | 0.06 (3%) | 1.20 (5%) |
| total | 1.76 | 21.90 |

During decode the projections and the output head read 799 MB in 18.2 ms, which is 44 GB/s, so the
matmuls themselves run at about what `infer membw` measures for the whole machine. The biggest piece
of what is left is the DeltaNet state, about 20 MB that every token reads and writes back.

## How it works

### The int8 path

Weights in q8 files are int8 with one f16 scale per 32 weights of a row. Activations get quantized
to int8 on the fly, also with one scale per 32 values, and the quantizer writes down the sum of each
block of 32 as well.

The first version stored weights row by row, so every 32 weight block of every row paid for its own
int to float conversion, scale multiply and horizontal sum. Now the converter stores rows in groups
of 16, interleaved 4 bytes at a time, so one 64 byte vector holds 4 weights from each of 16 rows.
One AVX-512 `vpdpbusd` multiplies that vector by 4 activation bytes broadcast to every lane and adds
into 16 int32 lanes, one per row (AVX-VNNI does the same in two 32 byte halves). Eight of them
finish a block for all 16 rows, and then one conversion and one fused multiply-add apply the 16 row
scales. `vpdpbusd` wants unsigned bytes on one side, so q8 weights are flipped to unsigned with an
xor (which adds 128) and the accumulator starts at minus 128 times the block sum of the activations,
which cancels it exactly. q4 weights are stored as unsigned nibbles with an offset of 8 and go
through the same instruction. AVX2 without VNNI uses `vpmaddubsw` with the usual sign trick.

On the 2 core VM (AVX-512) the new layout cut the time spent in weight projections during prefill
from 9.7 to 1.7 ms per token. Prefill went from 93 to 336 tokens per second and decode from 14.6 to
25.2.

### Same answer every time

Every int8 output is computed the same way no matter which code path computes it: an exact int32 dot
product per block, converted to float, multiplied by the product of the two scales, and added into a
single float accumulator with a fused multiply-add, block by block in order. The scalar kernel does
exactly that with `std::fma`, so the scalar, AVX2, AVX-VNNI and AVX-512 kernels return the same
bits. A test checks it with `memcmp` for every batch size from 1 to 14 tokens.

The rest of the engine follows the same rule. How many tokens share one pass over the weights, how
the rows are split across threads and whether a token arrives alone or in the middle of a long
prefill chunk never changes the arithmetic for that token. Attention splits keys into fixed chunks
of 256 and merges them in order, and the DeltaNet recurrence always runs one sequence, one head and
one block of 64 state columns at a time. So a request gets bit-identical logits whether it is
decoded alone or batched with seven others, and the server test checks that five concurrent HTTP
requests return exactly the text each one returns by itself.

### Gated DeltaNet

Per head, the layer keeps a 128 by 128 state matrix S. For each token it decays S by a learned gate,
reads the memory for the current key, writes back the difference between the value and what it read,
scaled by a second gate, and reads the output with the query:

```
S = a * S
S = S + b * k (v - S^T k)^T
o = S^T q
```

Before that there is a causal depthwise convolution over the q, k and v channels, and after it a
gated RMSNorm. Each column of S evolves independently of the others, which the prefill kernel uses
twice: it runs a whole sequence through one block of columns while that block sits in L1, and it
fuses the write pass of one token with the decay pass of the next, which halves the passes over the
state. The result is bit-identical to running the tokens one at a time, and a test checks that too.

Speculative decoding needs to undo rejected tokens. Attention layers just truncate their KV cache,
but a DeltaNet state is a running sum and cannot be truncated. Each sequence owns one bank of
convolution and recurrent state per draft position plus the live one (about 20 MB per bank for
Qwen3.5-0.8B). During verification the state after every draft position is written into a spare
bank, and rolling back to the last accepted token just makes that bank the live one, so a rejection
costs no copy.

### Multi-token prediction

Qwen3.5 checkpoints include one extra layer trained to predict the token after the next one. It
takes the final hidden state and the embedding of the token that was just sampled, normalizes both,
concatenates them, projects back to the model width, runs one gated attention layer with its own KV
cache and reuses the main output head. Feeding a draft back in produces the next draft, so drafts
can be chained. The output head has 248,320 rows, which would make each draft cost a third of a full
forward pass, so drafts only score the first 32,768 token ids. BPE hands out low ids to the most
frequent merges, so most good drafts are still in range, and the speculative decoding table above
shows the trade.

### Scheduling

Each engine step builds one batch: one row per decoding sequence (plus its drafts), then prefill
chunks from newer requests until the token budget (256 by default) is full, so a long prompt is
spread over several steps and the requests that are already generating keep producing a token every
step. The KV cache is a shared pool of 16 token blocks that sequences take as they grow, so short
and long requests share memory without reserving a full context each. Requests beyond the number of
parallel slots wait in a queue.

### Tokenizer

The pre-tokenizer patterns in `tokenizer.json` (the Qwen2 and Qwen3 pattern, the Qwen3.5 variant
that treats combining marks as part of a word, Llama 3 and GPT-2) are implemented as hand written
matchers instead of a regex engine. BPE uses a heap over adjacent pairs and caches words it has
already seen. The Unicode tables are generated by `tools/gen_unicode.py`.

## Two bugs from one assumption

I assumed Unicode was Unicode.

The first version of the tables came from Python's `unicodedata`, and 4,433 code points disagreed
with Hugging Face on whether they were letters, marks or numbers. Python 3.13 ships Unicode 15.1;
the regex engine inside tokenizers (Oniguruma) is on 16.0. Generating the tables from `unicodedata2`
16.0 fixed the classes.

NFC then broke the other way. A correct Unicode 16 normalizer disagreed with tokenizers on 61 to 106
strings, because the normalizer inside tokenizers ships older composition data than its own regex
engine. There is no single Unicode version to target, so the generator now asks the tokenizers
library which decompositions and compositions it knows and keeps only those. After that, all
1,132,064 strings matched for both models.

## Building

Needs CMake 3.20+ and a C++20 compiler. There are no other dependencies.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build
```

On Windows this builds with MSVC (default generator) and with MinGW gcc (add `-G Ninja`).

## Using it

Download a model, for example with `hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B`,
then:

```
build/infer info models/Qwen3.5-0.8B
build/infer convert models/Qwen3.5-0.8B models/qwen35-q8.infer --type q8
build/infer run models/qwen35-q8.infer --chat -p "Explain how a hash map works." --draft mtp --stats
build/infer chat models/qwen35-q8.infer
build/infer serve models/qwen35-q8.infer --port 8080 --parallel 4
```

The server speaks the OpenAI chat format, so any OpenAI client works against it:

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Name three prime numbers."}],"stream":true}'
```

`infer bench`, `infer perplexity`, `infer membw` and `infer logits` reproduce the tables above.
`tools/check.py` holds the comparisons against Hugging Face tokenizers and transformers.

## Tests

Four test binaries with 36 cases, run by ctest:

- `kernels_test`: f16 and bf16 conversion, activation quantization identical on every kernel set,
  weight round trips through the grouped layout, int8 matmul bit-identical across kernel sets and
  tile sizes, batch and thread partition invariance, the DeltaNet kernels against a double precision
  reference.
- `tokenizer_test`: UTF-8 decoding and repair, NFC, character classes, pre-tokenizer splits, BPE
  merge order, special tokens, the streaming decoder and both chat templates.
- `model_test`: tiny random Qwen3.5 and Qwen3 models. Chunked prefill against token by token
  decoding, thread count independence, batched against solo sequences, speculative greedy decoding
  token for token, a sampling distribution test for speculative sampling, stop strings.
- `server_test`: every endpoint, streaming against non streaming, concurrent requests against solo
  runs, malformed requests.

`tools/check.py` holds the slower comparisons against tokenizers and transformers shown above, plus
a generator for small random Hugging Face checkpoints so the logits comparison can run without
downloading anything.

CI runs Linux gcc and clang with warnings as errors (repeating the suite once per kernel set the
runner's CPU supports), the address plus undefined and the thread sanitizers, macOS on arm64 with
the scalar kernels, Windows with MSVC, and clang-format. I also run everything on Windows with MinGW
gcc 15 and MSVC 19.44 on the laptop.

## License

MIT
