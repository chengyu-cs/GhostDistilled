"""Minimal OpenAI-compatible server for the local Qwen2.5 base/LoRA model.

  python3 serve_lora.py --adapter lora_out --port 8011
  OPENAI_BASE_URL=http://127.0.0.1:8011/v1 OPENAI_API_KEY=dummy \
    OPENAI_MODEL=local python3 ../persona_agent/conversation_test.py
"""
from __future__ import annotations

import argparse
import re
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LOCK = threading.Lock()
MODEL = None
TOKENIZER = None
MODEL_NAME = "local"


def load(base: str, adapter: str | None, load_4bit: bool = False) -> None:
    global MODEL, TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    kwargs = {
        "device_map": {"": 0},
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    MODEL = AutoModelForCausalLM.from_pretrained(base, low_cpu_mem_usage=True, **kwargs)
    if adapter:
        MODEL = PeftModel.from_pretrained(MODEL, adapter)
    MODEL.eval()
    print(f"loaded base={base} adapter={adapter or 'none'} 4bit={load_4bit}", flush=True)


def generate(payload: dict) -> dict:
    messages = payload.get("messages") or []
    max_tokens = int(payload.get("max_tokens") or 300)
    temperature = float(payload.get("temperature", 0.85))
    top_p = float(payload.get("top_p", 0.9))
    with LOCK, torch.inference_mode():
        try:
            prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = TOKENIZER(prompt, add_special_tokens=False, return_tensors="pt").to(MODEL.device)
        enc = batch["input_ids"]
        out = MODEL.generate(
            enc,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=TOKENIZER.eos_token_id,
        )
        text = TOKENIZER.decode(out[0][enc.shape[-1]:], skip_special_tokens=True).strip()
        # Qwen3.5 是思考型模型：滤掉泄漏的 <think> 块
        for tag in ("</think>", "Thinking Process:", "redacted_thinking"):
            if tag in text:
                text = text.split(tag)[-1].strip()
        text = re.sub(r"^\s*\n+", "", text)
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }




class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

    def _send(self, code: int, data: dict) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send(200, generate(payload))
        except Exception as exc:  # noqa: BLE001 - surface the error to the client
            import traceback; traceback.print_exc(); self._send(500, {"error": str(exc)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    load(args.base, args.adapter or None, load_4bit=args.load_4bit)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"listening on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
