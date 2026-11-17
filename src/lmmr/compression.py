"""Official LLMLingua-2 token selection as a retrieval-compression comparator."""
from pathlib import Path

MODEL_ID = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
MODEL_REVISION = "ebaba9b0e874dadd3003ffcff828e4397e568089"


class LLMLingua2Compressor:
    def __init__(self, model=MODEL_ID, device="cuda", revision=MODEL_REVISION):
        import torch
        from llmlingua import PromptCompressor
        settings = {"trust_remote_code": False,
                    "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32}
        if Path(model).is_dir():
            settings["local_files_only"] = True
        elif revision:
            settings["revision"] = revision
        self.engine = PromptCompressor(
            model_name=model, device_map=device, use_llmlingua2=True,
            model_config=settings,
            llmlingua2_config={"max_batch_size": 16, "max_force_token": 100})
        if self.engine.model.config.model_type != "xlm-roberta":
            raise ValueError("This comparator uses the official XLM-R Large checkpoint")
        # The official word-boundary utilities dispatch on the architecture name.
        self.engine.model_name = MODEL_ID
        self.engine.model.eval()
        self.torch = torch
        self.model_source = str(model)
        self.revision = None if Path(model).is_dir() else revision

    def compress(self, evidence, count_tokens, budget=400, rate=.75, prompt_fits=None):
        if budget <= 0 or not 0 < rate <= 1:
            raise ValueError("Use a positive budget and a retention rate in (0, 1]")
        actual_rate = rate
        for _ in range(5):
            with self.torch.inference_mode():
                result = self.engine.compress_prompt_llmlingua2(
                    [evidence], rate=actual_rate,
                    use_context_level_filter=False, use_token_level_filter=True,
                    force_tokens=[], return_word_label=False)
            text = result["compressed_prompt"]
            tokens = count_tokens(text)
            if tokens <= budget and (prompt_fits is None or prompt_fits(text)):
                return {"evidence": text, "evidence_tokens": tokens,
                        "requested_rate": actual_rate,
                        "model_family": MODEL_ID, "model_source": self.model_source,
                        "revision": self.revision}
            actual_rate *= min(.95, max(1, budget - 10) / max(tokens, 1))
        raise ValueError("Compressed evidence exceeds the reference tokenizer/prompt budget")
