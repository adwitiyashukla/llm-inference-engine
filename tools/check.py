import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SAMPLE_TEXT = (
    "The old harbor town wakes up slowly. Before sunrise the fishing boats come back with their "
    "lights still on, and the gulls follow them past the breakwater. By seven the bakery on the "
    "square has opened its shutters, the bus to the university leaves from the stop by the clock "
    "tower, and students argue about exams over coffee that is too strong. In the afternoon the "
    "wind turns, the tide pulls out past the rocks, and children search the pools for crabs. "
    "Nobody here is in a hurry, but the town still keeps its own schedule: the ferry at noon, the "
    "market on Thursdays, the brass band rehearsing badly every Sunday evening. Visitors often say "
    "that nothing happens in a place like this. The people who live here know better. "
)

WORDS = [
    "hello", "world", "the", "model", "token", "cache", "think", "answer", "Qwen", "batch",
    "don't", "it's", "WE'LL", "I'm", "naive", "caf\u00e9", "na\u0308ive", "\u4e2d\u6587",
    "\u65e5\u672c\u8a9e", "\ud55c\uad6d\uc5b4", "\u0645\u0631\u062d\u0628\u0627", "\u0928\u092e\u0938\u094d\u0924\u0947",
    "\U0001f600", "\U0001f44d\U0001f3fd", "123", "4567", "3.14", "x^2", "<b>", "</b>", "{\"a\": 1}",
    "def f(x):", "return x", "#include", "\u00a0", "\u3000", "\u200b", "\u2163", "\u2126", "\u0958",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_", "<think>", "</think>", "\u1100\u1161\u11a8",
]

SPACES = [" ", "  ", "\n", "\n\n", "\t", " \n", "\r\n", "   "]


def infer_bin(args):
    p = Path(args.infer)
    if os.name == "nt" and p.suffix == "" and not p.exists():
        p = p.with_suffix(".exe")
    return str(p)


def run_infer(args, *cmd):
    extra = ["--threads", str(args.threads)] if getattr(args, "threads", None) else []
    proc = subprocess.run([infer_bin(args), *map(str, cmd), *extra], capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"infer {cmd[0]} failed with exit code {proc.returncode}")
    return proc.stdout


def target(args):
    return args.model or args.model_dir


def write_records(path, texts):
    with open(path, "wb") as f:
        for t in texts:
            b = t.encode("utf-8")
            f.write(f"{len(b)}\n".encode())
            f.write(b)


def records_output(args, path, texts, *flags):
    write_records(path, texts)
    out = run_infer(args, "tokenize", path.parent / "tokenizer.json", "--records", path, *flags)
    lines = out.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(texts):
        raise SystemExit(f"expected {len(texts)} records back, got {len(lines)}")
    return lines


def random_text(rng):
    parts = []
    for _ in range(rng.randint(1, 12)):
        r = rng.random()
        if r < 0.45:
            parts.append(rng.choice(WORDS))
        elif r < 0.7:
            parts.append(rng.choice(SPACES))
        elif r < 0.85:
            parts.append("".join(chr(rng.randint(32, 126)) for _ in range(rng.randint(1, 6))))
        else:
            cp = rng.randint(0xA0, 0x2FFFF)
            if 0xD800 <= cp <= 0xDFFF:
                cp = 0x4E00
            parts.append(chr(cp) * rng.randint(1, 3))
    return "".join(parts)


def report(name, total, bad, examples):
    print(f"{name}: {total - bad}/{total} match, {bad} mismatches")
    for e in examples[:5]:
        print("  " + e)
    return bad


def cmd_tokenizer(args):
    from tokenizers import Tokenizer, normalizers

    src = Path(args.model_dir) / "tokenizer.json"
    hf = Tokenizer.from_file(str(src))
    texts = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        texts.append("a" + c + "b " + c + c + "'s 1" + c)
    rng = random.Random(args.seed)
    texts += [random_text(rng) for _ in range(args.random)]
    if args.corpus:
        with open(args.corpus, encoding="utf-8", newline="") as f:
            texts += [p for p in f.read().split("\n\n") if p]
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(src, Path(d) / "tokenizer.json")
        rec = Path(d) / "records.bin"
        t0 = time.perf_counter()
        ours = records_output(args, rec, texts)
        t_ours = time.perf_counter() - t0
        t0 = time.perf_counter()
        theirs = hf.encode_batch(texts, add_special_tokens=False)
        t_hf = time.perf_counter() - t0
        bad = []
        tokens = 0
        for text, line, enc in zip(texts, ours, theirs):
            ref = " ".join(map(str, enc.ids))
            tokens += len(enc.ids)
            if line != ref:
                bad.append(f"{text!r}: ours [{line}] reference [{ref}]")
        failures += report("encode", len(texts), len(bad), bad)
        print(f"  {tokens} tokens; infer tokenize {t_ours:.1f}s (one thread, includes startup), "
              f"tokenizers encode_batch {t_hf:.1f}s")
        if hf.normalizer is not None:
            nfc = normalizers.NFC()
            ours = records_output(args, rec, texts, "--mode", "nfc")
            bad = []
            for text, line in zip(texts, ours):
                ref = nfc.normalize_str(text).encode("utf-8").hex()
                if line != ref:
                    bad.append(f"{text!r}: ours {line} reference {ref}")
            failures += report("nfc", len(texts), len(bad), bad)
    raise SystemExit(1 if failures else 0)


def random_content(rng, allow_think):
    text = "".join(random_text(rng) for _ in range(rng.randint(1, 3)))
    if rng.random() < 0.3:
        text = rng.choice(SPACES) + text + rng.choice(SPACES)
    if allow_think and rng.random() < 0.4:
        inner = random_text(rng)
        open_tag = "<think>" + rng.choice(["", "\n", "\n\n"]) if rng.random() < 0.8 else ""
        text = open_tag + inner + rng.choice(["", "\n", "\n\n"]) + "</think>" + rng.choice(["", "\n", "\n\n"]) + text
    return text


def random_conversation(rng):
    msgs = []
    if rng.random() < 0.3:
        msgs.append({"role": "system", "content": random_content(rng, False)})
    role = "user"
    for _ in range(rng.randint(1, 6)):
        r = rng.random()
        if r < 0.04:
            msgs.append({"role": "system", "content": random_content(rng, False)})
            continue
        if role == "user":
            content = random_content(rng, False)
            if rng.random() < 0.12:
                content = "<tool_response>" + content + "</tool_response>"
            msgs.append({"role": "user", "content": content})
            role = "assistant"
        else:
            m = {"role": "assistant", "content": random_content(rng, True)}
            if rng.random() < 0.25:
                m["reasoning_content"] = random_text(rng)
            msgs.append(m)
            role = "user" if rng.random() < 0.9 else "assistant"
    return msgs


def cmd_chat(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    rng = random.Random(args.seed)
    cases = []
    for _ in range(args.count):
        cases.append({"messages": random_conversation(rng), "add": rng.random() < 0.7,
                      "thinking": rng.choice([-1, 0, 1]), "format": args.format})
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(Path(args.model_dir) / "tokenizer.json", Path(d) / "tokenizer.json")
        ours = records_output(args, Path(d) / "chat.bin", [json.dumps(c) for c in cases],
                              "--mode", "chat", "--format", args.format)
    bad = []
    errors = 0
    for case, line in zip(cases, ours):
        kw = {}
        if case["thinking"] >= 0:
            kw["enable_thinking"] = bool(case["thinking"])
        try:
            ref = tok.apply_chat_template(case["messages"], tokenize=False,
                                          add_generation_prompt=case["add"], **kw)
            ref = ref.encode("utf-8").hex()
        except Exception:
            ref = "ERROR"
            errors += 1
        if line != ref:
            got = bytes.fromhex(line).decode("utf-8") if line != "ERROR" else line
            want = bytes.fromhex(ref).decode("utf-8") if ref != "ERROR" else ref
            bad.append(f"{json.dumps(case)}\n    ours {got!r}\n    reference {want!r}")
    n = report("chat template", len(cases), len(bad), bad)
    print(f"  {errors} conversations rejected by the reference template")
    raise SystemExit(1 if n else 0)


def load_hf(model_dir, dtype):
    import torch
    import transformers

    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity_error()
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    kind = cfg.get("model_type")
    if kind == "qwen3_5" and "text_config" in cfg:
        cls = getattr(transformers, "AutoModelForImageTextToText")
    else:
        cls = getattr(transformers, "AutoModelForCausalLM")
    model = cls.from_pretrained(model_dir, dtype=dtype)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def sample_ids(args, tok):
    ids = tok(SAMPLE_TEXT * args.repeat, add_special_tokens=False)["input_ids"]
    return ids[: args.tokens]


def softmax64(x):
    import numpy as np

    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cmd_logits(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = sample_ids(args, tok)
    with tempfile.TemporaryDirectory() as d:
        idf = Path(d) / "ids.txt"
        idf.write_text(" ".join(map(str, ids)))
        out = Path(d) / "logits.bin"
        run_infer(args, "logits", target(args), "--ids-file", idf, "--out", out, "--kv", args.kv)
        ours = np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)
    model = load_hf(args.model_dir, torch.float32)
    ref = model(input_ids=torch.tensor([ids])).logits[0].float().numpy()
    del model
    v = min(ref.shape[1], ours.shape[1])
    ref = ref[:, :v]
    ours = ours[:, :v]
    diff = np.abs(ref - ours)
    top1 = float((ref.argmax(-1) == ours.argmax(-1)).mean())
    p = softmax64(ref)
    q = softmax64(ours)
    kl = float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1).mean())
    print(f"logits over {len(ids)} positions, vocab {v}, kv {args.kv}")
    print(f"  max abs diff {diff.max():.4g}, mean abs diff {diff.mean():.4g}")
    print(f"  top-1 agreement {100 * top1:.2f}%, mean KL(reference || infer) {kl:.3g}")
    raise SystemExit(0 if top1 >= args.min_agreement else 1)


GREEDY_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a short Python function that reverses a string.",
    "List three facts about the planet Mars.",
    "What is 17 times 23? Show the steps.",
]


def cmd_greedy(args):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    prompts = []
    for p in GREEDY_PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok(text, add_special_tokens=False)["input_ids"])
    ours = []
    with tempfile.TemporaryDirectory() as d:
        for i, ids in enumerate(prompts):
            idf = Path(d) / f"ids{i}.txt"
            idf.write_text(" ".join(map(str, ids)))
            extra = ["--draft", args.draft] if args.draft else []
            line = run_infer(args, "logits", target(args), "--ids-file", idf, "--greedy", args.new,
                             "--kv", args.kv, *extra).decode().split()
            ours.append([int(x) for x in line])
    model = load_hf(args.model_dir, torch.float32)
    total = 0
    matched = 0
    for p, ids in enumerate(prompts):
        out = model(input_ids=torch.tensor([ids]), use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        ref = []
        gaps = []
        for _ in range(args.new):
            top = torch.topk(logits.float(), 2)
            ref.append(int(top.indices[0]))
            gaps.append(float(top.values[0] - top.values[1]))
            out = model(input_ids=torch.tensor([[ref[-1]]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
        same = 0
        while same < len(ref) and same < len(ours[p]) and ref[same] == ours[p][same]:
            same += 1
        total += len(ref)
        matched += same
        note = "" if same == len(ref) else f", first difference at {same} (reference top-2 gap {gaps[same]:.4f})"
        print(f"prompt {p}: {same}/{len(ref)} tokens identical{note}")
        print("  " + repr(tok.decode(ours[p][: max(same, 1) + 8])))
    print(f"greedy: {matched}/{total} tokens identical across {len(prompts)} prompts")
    raise SystemExit(0 if matched == total else 1)


def cmd_ppl(args):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    with open(args.file, encoding="utf-8", newline="") as f:
        ids = tok(f.read(), add_special_tokens=False)["input_ids"]
    windows = len(ids) // args.ctx
    if args.chunks:
        windows = min(windows, args.chunks)
    if args.threads:
        torch.set_num_threads(args.threads)
    model = load_hf(args.model_dir, torch.float32)
    nll = 0.0
    scored = 0
    t0 = time.perf_counter()
    for w in range(windows):
        x = torch.tensor([ids[w * args.ctx:(w + 1) * args.ctx]])
        logits = model(input_ids=x).logits[0]
        for r in range(0, args.ctx - 1, 64):
            part = logits[r:min(r + 64, args.ctx - 1)].double()
            lp = torch.log_softmax(part, -1)
            tgt = x[0, r + 1:r + 1 + part.shape[0]].unsqueeze(1)
            nll -= float(lp.gather(1, tgt).sum())
        scored += args.ctx - 1
        print(f"[{w + 1}/{windows}] ppl {math.exp(nll / scored):.4f}", file=sys.stderr)
    secs = time.perf_counter() - t0
    print(f"transformers perplexity {math.exp(nll / scored):.4f} over {scored} tokens "
          f"({windows} windows of {args.ctx}), {windows * args.ctx / secs:.1f} tok/s")


def cmd_speed(args):
    import torch
    from transformers import AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = sample_ids(argparse.Namespace(repeat=8, tokens=args.prompt), tok)
    x = torch.tensor([ids])
    for name in args.dtypes.split(","):
        dtype = getattr(torch, name)
        model = load_hf(args.model_dir, dtype)
        gen = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
                   pad_token_id=tok.eos_token_id or 0)
        model.generate(x, max_new_tokens=4, min_new_tokens=4, **gen)
        t0 = time.perf_counter()
        model.generate(x, max_new_tokens=1, min_new_tokens=1, **gen)
        t1 = time.perf_counter()
        model.generate(x, max_new_tokens=args.gen, min_new_tokens=args.gen, **gen)
        t2 = time.perf_counter()
        prefill = (t1 - t0)
        decode = (t2 - t1) - prefill
        print(f"pytorch {torch.__version__} {name}: prefill {len(ids) / prefill:.1f} tok/s, "
              f"decode {(args.gen - 1) / decode:.2f} tok/s, threads {torch.get_num_threads()}")
        del model


def cmd_random(args):
    import torch
    import transformers
    from safetensors.torch import load_file, save_file

    transformers.utils.logging.disable_progress_bar()
    torch.manual_seed(args.seed)
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.tokenizer_from)
    vocab = len(json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]) + 64
    vocab = (vocab + 127) // 128 * 128
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        cfg = cfg.get("text_config", cfg)
    elif args.arch == "qwen3":
        cfg = dict(model_type="qwen3", hidden_size=128, intermediate_size=384, num_hidden_layers=3,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                   rope_theta=1000000.0, tie_word_embeddings=True, vocab_size=vocab)
    else:
        cfg = dict(model_type="qwen3_5_text", hidden_size=128, intermediate_size=384,
                   num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                   layer_types=["linear_attention", "linear_attention", "linear_attention",
                                "full_attention"],
                   linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                   linear_value_head_dim=32, linear_conv_kernel_dim=4,
                   rope_parameters=dict(rope_type="default", rope_theta=10000000.0,
                                        partial_rotary_factor=0.25),
                   tie_word_embeddings=True, vocab_size=vocab)
    cfg = dict(cfg)
    cfg["vocab_size"] = max(cfg.get("vocab_size", vocab), vocab)
    kind = cfg.pop("model_type")
    mtp_layers = cfg.pop("mtp_num_hidden_layers", 0)
    if kind == "qwen3":
        conf = transformers.Qwen3Config(**cfg)
        model = transformers.Qwen3ForCausalLM(conf)
    else:
        conf = transformers.Qwen3_5TextConfig(**cfg)
        model = transformers.Qwen3_5ForCausalLM(conf)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, args.std)
            elif "norm" in name and kind != "qwen3":
                p.normal_(0.0, 0.1)
            elif "norm" in name:
                p.normal_(1.0, 0.1)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json",
              "merges.txt", "generation_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    if args.mtp and kind != "qwen3":
        h = conf.hidden_size
        hd = conf.head_dim
        g = torch.Generator().manual_seed(args.seed + 1)

        def mat(r, c):
            return (torch.randn(r, c, generator=g) * args.std).to(torch.bfloat16)

        def vec(n):
            return (torch.randn(n, generator=g) * 0.1).to(torch.bfloat16)

        p = "mtp.layers.0."
        extra = {
            "mtp.fc.weight": mat(h, 2 * h),
            "mtp.pre_fc_norm_embedding.weight": vec(h),
            "mtp.pre_fc_norm_hidden.weight": vec(h),
            "mtp.norm.weight": vec(h),
            p + "input_layernorm.weight": vec(h),
            p + "post_attention_layernorm.weight": vec(h),
            p + "self_attn.q_proj.weight": mat(2 * conf.num_attention_heads * hd, h),
            p + "self_attn.k_proj.weight": mat(conf.num_key_value_heads * hd, h),
            p + "self_attn.v_proj.weight": mat(conf.num_key_value_heads * hd, h),
            p + "self_attn.o_proj.weight": mat(h, conf.num_attention_heads * hd),
            p + "self_attn.q_norm.weight": vec(hd),
            p + "self_attn.k_norm.weight": vec(hd),
            p + "mlp.gate_proj.weight": mat(conf.intermediate_size, h),
            p + "mlp.up_proj.weight": mat(conf.intermediate_size, h),
            p + "mlp.down_proj.weight": mat(h, conf.intermediate_size),
        }
        st = out / "model.safetensors"
        tensors = load_file(str(st))
        tensors.update(extra)
        save_file(tensors, str(st), metadata={"format": "pt"})
        mtp_layers = 1
    if mtp_layers:
        c = json.loads((out / "config.json").read_text(encoding="utf-8"))
        c["mtp_num_hidden_layers"] = mtp_layers
        (out / "config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    params = sum(p.numel() for p in model.parameters())
    print(f"wrote random {kind} model with {params / 1e6:.1f}M parameters to {out}")


def main():
    ap = argparse.ArgumentParser(description="check infer against Hugging Face tokenizers and transformers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        p = sub.add_parser(name)
        p.add_argument("model_dir")
        p.add_argument("--infer", default="build/infer")
        p.add_argument("--model", default=None)
        p.add_argument("--kv", default="f32", choices=["f16", "f32"])
        p.add_argument("--threads", type=int, default=None)
        p.set_defaults(fn=fn)
        return p

    p = add("tokenizer", cmd_tokenizer)
    p.add_argument("--random", type=int, default=20000)
    p.add_argument("--corpus", default=None)
    p.add_argument("--seed", type=int, default=1)
    p = add("chat", cmd_chat)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--format", default="qwen35", choices=["qwen3", "qwen35"])
    p.add_argument("--seed", type=int, default=2)
    p = add("logits", cmd_logits)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--min-agreement", type=float, default=0.99)
    p = add("greedy", cmd_greedy)
    p.add_argument("--new", type=int, default=64)
    p.add_argument("--draft", default=None, choices=["mtp", "ngram"])
    p = add("ppl", cmd_ppl)
    p.add_argument("--file", required=True)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--chunks", type=int, default=0)
    p = add("speed", cmd_speed)
    p.add_argument("--prompt", type=int, default=128)
    p.add_argument("--gen", type=int, default=32)
    p.add_argument("--dtypes", default="bfloat16,float32")
    p = add("random", cmd_random)
    p.add_argument("--arch", default="qwen35", choices=["qwen3", "qwen35"])
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer-from", required=True)
    p.add_argument("--mtp", action="store_true")
    p.add_argument("--std", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
