import asyncio
import logging
import os
import time
import json
import re
from enum import Enum
from typing import Optional, Dict, Any

# Conditional import for Llama
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

log = logging.getLogger(__name__)

class ModelProvider(Enum):
    LOCAL = "local"
    GROQ = "groq"

class LLMManager:
    """
    Manages a pool of Llama models and orchestrates inference requests.
    Centralizes prompt formatting, thread management, and JSON repair.
    """
    def __init__(self):
        self._model_pool: asyncio.Queue = asyncio.Queue()
        self.model_loaded = False
        self.instance_count = 0

    def load_local_models(self, model_path: str, max_instances: int = 1, context_size: int = 8192, machine_type: str = "mac"):
        """Preloads N instances of the model into the async pool."""
        if Llama is None:
            log.error("❌ llama-cpp-python not installed.")
            return

        if not os.path.exists(model_path):
            log.error(f"❌ Model file not found: {model_path}")
            return

        log.info(f"⏳ Initializing Model Pool with {max_instances} instances...")
        t0 = time.time()

        try:
            for i in range(max_instances):
                log.info(f"   ... Loading instance {i+1}/{max_instances}")
                # Configuration logic
                if machine_type == "mac":
                    model = Llama(
                        model_path=model_path, n_ctx=context_size, n_threads=4,
                        n_gpu_layers=-1, verbose=False, flash_attn=True
                    )
                else:
                    model = Llama(
                        model_path=model_path, n_ctx=4096, n_threads=6,
                        n_gpu_layers=0, verbose=False
                    )
                self._model_pool.put_nowait(model)
            
            self.model_loaded = True
            self.instance_count = max_instances
            log.info(f"✅ Model Pool Ready: {max_instances} instances loaded in {time.time() - t0:.2f}s")
            
        except Exception as e:
            log.error(f"❌ Failed to load models: {str(e)}")

    async def request_inference(
        self, 
        system_prompt: str, 
        user_text: str, 
        doc_label: str = "DOCUMENT", 
        is_private: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Acquires a model, runs inference, and returns cleaned JSON.
        """
        # 1. Provider Logic (Future-proof hook for Groq)
        provider = ModelProvider.LOCAL 
        
        full_prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

### {doc_label}:
{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        raw_text = ""
        
        if provider == ModelProvider.LOCAL:
            if not self.model_loaded:
                raise RuntimeError("Local models are not loaded.")
            
            # 2. CHECKOUT
            model_instance = await self._model_pool.get()
            
            try:
                # 3. PROCESS (Offload to thread)
                raw_text = await asyncio.to_thread(
                    self._run_local_sync, 
                    model_instance, 
                    full_prompt, 
                    temperature, 
                    max_tokens
                )
            finally:
                # 4. CHECKIN
                self._model_pool.put_nowait(model_instance)

        return self._clean_and_parse_json(raw_text)

    def _run_local_sync(self, model: Llama, prompt: str, temperature: float, max_tokens: int) -> str:
        """Internal sync call."""
        output = model(
            prompt,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
            temperature=temperature,
            echo=False,
            repeat_penalty=1.05,
            top_p=0.95
        )
        return output['choices'][0]['text'].strip()

    def _clean_and_parse_json(self, result_text: str) -> Dict[str, Any]:
        """
        Implements the ROBUST cleanup logic you requested.
        """
        # 1. Robust Markdown Cleanup
        if "```" in result_text:
            parts = result_text.split("```")
            found_json = False
            for p in parts:
                clean_p = p.strip()
                if clean_p.lower().startswith("json"):
                    clean_p = clean_p[4:].strip()
                if clean_p.startswith("{"):
                    result_text = clean_p
                    found_json = True
                    break
            
            if not found_json:
                 for p in reversed(parts):
                     if p.strip().startswith("{"):
                         result_text = p.strip()
                         break

        result_text = result_text.strip()
        
        start_idx = result_text.find("{")
        if start_idx != -1:
            result_text = result_text[start_idx:]

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            log.warning("⚠️ JSON Decode Error. Attempting repair...")
            fixed_text = self._repair_json(result_text)
            try:
                return json.loads(fixed_text)
            except:
                log.error("❌ Failed to repair JSON.")
                return {}

    def _repair_json(self, json_str: str) -> str:
        """Your specific repair logic."""
        json_str = json_str.strip()
        # 1. Fix missing commas between key-value pairs
        json_str = re.sub(r'\"\s*\n\s*\"', '",\n"', json_str)
        # 2. Close unclosed string quotes
        if json_str.count('"') % 2 != 0: json_str += '"'
        # 3. Remove trailing commas
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        # 4. Bruteforce closure
        closers = ["}", "]}", ""]
        for c in closers:
            try:
                candidate = json_str + c
                json.loads(candidate)
                return candidate
            except:
                continue
        if not json_str.endswith("}"): return json_str + "]}"
        return json_str


