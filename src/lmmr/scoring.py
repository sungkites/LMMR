"""Qwen reranker scores and native query/value LoRA adapters."""
from .memory import PREFIX, SUFFIX, STANDARD, CONDITIONAL


class QwenScorer:
    def __init__(self, model_path, adapter=None, device=None, batch_size=16, trainable=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.bfloat16 if trainable and self.device.startswith("cuda") else (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="sdpa").to(self.device)
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.batch_size = batch_size
        self.adapters = []
        yes = self.tokenizer.encode("yes", add_special_tokens=False)
        no = self.tokenizer.encode("no", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("The scorer requires single-token yes/no labels")
        self.output_weight = (self.model.lm_head.weight[yes[0]].float() - self.model.lm_head.weight[no[0]].float()).detach()
        if trainable:
            self._attach_lora()
        if adapter:
            state = torch.load(adapter, map_location=self.device, weights_only=True)
            if trainable:
                for name, layer in self.adapters:
                    layer.a.data.copy_(state[name + ".a"])
                    layer.b.data.copy_(state[name + ".b"])
            else:
                self.merge_adapter(adapter)
        self.model.eval()

    def merge_adapter(self, path):
        if self.adapters:
            raise ValueError('Do not merge weights into a trainable scorer')
        state = self.torch.load(path, map_location=self.device, weights_only=True)
        matched = set()
        with self.torch.no_grad():
            for name, layer in self.model.named_modules():
                if name + '.a' in state:
                    if name + '.b' not in state or not hasattr(layer, 'weight'):
                        raise ValueError('Incomplete adapter: ' + name)
                    delta = 2 * state[name + '.b'].float() @ state[name + '.a'].float()
                    layer.weight.add_(delta.to(layer.weight.dtype))
                    matched.update((name + '.a', name + '.b'))
        if matched != set(state) or not matched:
            raise ValueError('Checkpoint keys do not match the reranker')

    def _attach_lora(self):
        torch = self.torch
        import math
        class LoRALinear(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.a = torch.nn.Parameter(torch.empty(8, base.in_features, device=base.weight.device, dtype=torch.float32))
                self.b = torch.nn.Parameter(torch.zeros(base.out_features, 8, device=base.weight.device, dtype=torch.float32))
                torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
            def forward(self, x):
                import torch.nn.functional as F
                return self.base(x) + (2 * F.linear(F.linear(x.float(), self.a), self.b)).to(x.dtype)
        for name, layer in list(self.model.named_modules()):
            if name.endswith((".q_proj", ".v_proj")) and isinstance(layer, torch.nn.Linear):
                parent, key = name.rsplit(".", 1)
                adapted = LoRALinear(layer)
                setattr(self.model.get_submodule(parent), key, adapted)
                self.adapters.append((name, adapted))

    def adapter_parameters(self):
        return [p for _, layer in self.adapters for p in (layer.a, layer.b)]

    def save_adapter(self, path):
        self.torch.save({name + suffix: p.detach().cpu() for name, layer in self.adapters
                         for suffix, p in [(".a", layer.a), (".b", layer.b)]}, path)

    def forward(self, records):
        sequences = []
        for question, candidate, anchor, conditional in records:
            query = question if anchor is None else question + "\nKnown evidence:\n" + anchor
            instruction = CONDITIONAL if conditional else STANDARD
            prompt = PREFIX + "<Instruct>: " + instruction + "\n<Query>: " + query + "\n<Document>: " + candidate + SUFFIX
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > 1536:
                raise ValueError("Reranking input exceeds 1536 tokens")
            sequences.append(ids)
        encoded = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(self.device)
        lengths = encoded["attention_mask"].sum(-1) - 1
        hidden = self.model.model(**encoded, use_cache=False).last_hidden_state
        last = hidden[self.torch.arange(len(records), device=self.device), lengths]
        return (last.float() * self.output_weight).sum(-1)

    def score(self, question, candidates, anchor=None, conditional=False):
        output = []
        with self.torch.inference_mode():
            for i in range(0, len(candidates), self.batch_size):
                records = [(question, n["text"], anchor, conditional) for n in candidates[i:i+self.batch_size]]
                output.extend(self.forward(records).cpu().tolist())
        return output
