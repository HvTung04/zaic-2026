import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM:
    def __init__(self, model_id="Qwen/Qwen3-1.7B", device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map=device, local_files_only=True
        )

    def generate(self, messages, max_new_tokens=512):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        if inputs.input_ids.shape[1] > 14000:
            # crude truncation of oldest content by rebuilding not supported; rely on caller
            pass
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    @staticmethod
    def parse_calls(raw):
        raw = raw.strip()
        # extract first JSON array
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        calls = []
        for c in data:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            args = c.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            if isinstance(name, str):
                calls.append({"name": name, "arguments": args})
        return calls