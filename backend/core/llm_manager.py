import asyncio
import os
import time
import json
import re
from enum import Enum
from typing import Optional, Dict, Any, Union, List

# Conditional import for Llama
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class ModelProvider(Enum):
    LOCAL = "local"
    GROQ = "groq"

class BespokeJSONRecovery:
    """
    Acts as a final safety net. If JSON cannot be parsed or repaired,
    this class uses regex to hunt for expected attributes based on the prompt type.
    """
    
    @staticmethod
    def recover(raw_text: str, expected_type: str) -> Union[Dict[str, Any], List[Any]]:
        """Routes the broken text to the correct regex recovery template."""
        print(f"\n⚠️ [STEP 4: BESPOKE RECOVERY] Initiating regex recovery for type: {expected_type}\n")
        
        if expected_type == "JOB_DESCRIPTION":
            return BespokeJSONRecovery._recover_job_profile(raw_text)
        elif expected_type in ["CV_EXPERIENCE", "CV_EDUCATION", "CV_PROJECTS", "CV_HOBBIES"]:
            return BespokeJSONRecovery._recover_cv_items(raw_text)
        elif expected_type == "CV_IDENTITY":
            return BespokeJSONRecovery._recover_cv_identity(raw_text)
        
        return {}

    @staticmethod
    def _extract_string(pattern: str, text: str) -> Optional[str]:
        """Helper to safely extract a single regex match."""
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # Clean up trailing quotes or commas that regex might have caught
            return re.sub(r'["\',]+$', '', extracted).strip()
        return None

    @staticmethod
    def _recover_job_profile(text: str) -> Dict[str, Any]:
        """Recovers Job Description flat fields and attempts to find features."""
        recovered = {
            "title": BespokeJSONRecovery._extract_string(r'"title"\s*:\s*"([^"]+)"', text),
            "company": BespokeJSONRecovery._extract_string(r'"company"\s*:\s*"([^"]+)"', text),
            "location": BespokeJSONRecovery._extract_string(r'"location"\s*:\s*"([^"]+)"', text),
            "salary_range": BespokeJSONRecovery._extract_string(r'"salary_range"\s*:\s*"([^"]+)"', text),
            "features": []
        }
        
        # Hunt for feature objects specifically
        feature_blocks = re.findall(r'\{\s*"type"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]+)"\s*\}', text)
        for f_type, f_desc in feature_blocks:
            recovered["features"].append({
                "type": f_type.strip(),
                "description": f_desc.strip()
            })
            
        return recovered

    @staticmethod
    def _recover_cv_items(text: str) -> Dict[str, Any]:
        """Recovers array-based CV sections by finding object blocks."""
        recovered = {"items": []}
        
        # Find everything that looks like an object block inside the text
        object_blocks = re.findall(r'\{([^{}]+)\}', text)
        
        for block in object_blocks:
            item = {}
            # Extract common CV fields from the broken block
            keys_to_hunt = [
                "company", "title", "start_date", "end_date", "location",
                "description", "achievements_list", "tools_used",
                "institution", "qualification", "field", "grade", "details_list",
                "name", "related_context"
            ]
            
            for key in keys_to_hunt:
                val = BespokeJSONRecovery._extract_string(rf'"{key}"\s*:\s*"([^"]+)"', block)
                if val and val.lower() != "null":
                    item[key] = val
                    
            if item:
                recovered["items"].append(item)
                
        return recovered

    @staticmethod
    def _recover_cv_identity(text: str) -> Dict[str, Any]:
        """Recovers CV Identity fields and skills."""
        recovered = {
            "first_name": BespokeJSONRecovery._extract_string(r'"first_name"\s*:\s*"([^"]+)"', text),
            "last_name": BespokeJSONRecovery._extract_string(r'"last_name"\s*:\s*"([^"]+)"', text),
            "email": BespokeJSONRecovery._extract_string(r'"email"\s*:\s*"([^"]+)"', text),
            "phone": BespokeJSONRecovery._extract_string(r'"phone"\s*:\s*"([^"]+)"', text),
            "linkedin": BespokeJSONRecovery._extract_string(r'"linkedin"\s*:\s*"([^"]+)"', text),
            "location": BespokeJSONRecovery._extract_string(r'"location"\s*:\s*"([^"]+)"', text),
            "summary": BespokeJSONRecovery._extract_string(r'"summary"\s*:\s*"([^"]+)"', text),
            "skills": []
        }
        
        skill_blocks = re.findall(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"category"\s*:\s*"([^"]+)"\s*\}', text)
        for s_name, s_cat in skill_blocks:
            recovered["skills"].append({
                "name": s_name.strip(),
                "category": s_cat.strip()
            })
            
        return recovered


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
            print("\n❌ [ERROR] llama-cpp-python not installed.\n")
            return

        if not os.path.exists(model_path):
            print(f"\n❌ [ERROR] Model file not found: {model_path}\n")
            return

        print(f"\n⏳ Initializing Model Pool with {max_instances} instances...\n")
        t0 = time.time()

        try:
            for i in range(max_instances):
                print(f"   ... Loading instance {i+1}/{max_instances}")
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
            print(f"\n✅ Model Pool Ready: {max_instances} instances loaded in {time.time() - t0:.2f}s\n")
            
        except Exception as e:
            print(f"\n❌ [ERROR] Failed to load models: {str(e)}\n")

    async def request_inference(
        self, 
        system_prompt: str, 
        user_text: str, 
        doc_label: str = "DOCUMENT", 
        is_private: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Acquires a model, runs inference, and returns cleaned JSON.
        """
        # Automatically infer expected_type based on prompt persona strings
        expected_type = "UNKNOWN"
        if "Lead HR Analyst" in system_prompt:
            expected_type = "JOB_DESCRIPTION"
        elif "Lead CV Analyst" in system_prompt:
            expected_type = "CV_IDENTITY"
        elif "Work History Auditor" in system_prompt:
            expected_type = "CV_EXPERIENCE"
        elif "Academic Researcher" in system_prompt:
            expected_type = "CV_EDUCATION"
        elif "Portfolio Analyst" in system_prompt:
            expected_type = "CV_PROJECTS"
        elif "Extracurricular Auditor" in system_prompt:
            expected_type = "CV_HOBBIES"

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

        print(f"\n🤖 [RAW LLM OUTPUT]\n{raw_text}\n🤖 [END RAW OUTPUT]\n")

        return self._clean_and_parse_json(raw_text, expected_type)

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

    def _clean_and_parse_json(self, result_text: str, expected_type: str = "UNKNOWN") -> Union[Dict[str, Any], List[Any]]:
        """
        Robustly extracts the largest JSON object or array. Follows a progressive fallback logic.
        """
        print("\n🔍 [STEP 1] Trying to parse raw text immediately...\n")
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            print("\n🧹 [STEP 2] Raw parse failed. Cleaning text and hunting for brackets...\n")
            pass 

        # STEP 2: Clean and parse
        # Strip markdown code blocks explicitly
        clean_text = re.sub(r'```(?:json)?', '', result_text).strip()
        clean_text = clean_text.strip('`').strip()

        json_candidate = ""
        
        # 1. Find the first JSON opening character
        start_idx = -1
        for i, char in enumerate(clean_text):
            if char in ['{', '[']:
                start_idx = i
                break
                
        # 2. Extract from the first opener to the last matching closer
        if start_idx != -1:
            opening_char = clean_text[start_idx]
            closing_char = ']' if opening_char == '[' else '}'
            
            end_idx = clean_text.rfind(closing_char)
            
            if end_idx != -1 and end_idx >= start_idx:
                json_candidate = clean_text[start_idx:end_idx+1]
            else:
                # Fallback if text is severely truncated
                json_candidate = clean_text[start_idx:] 
        else:
            print("\n⚠️ [WARNING] No JSON braces found. Routing directly to bespoke recovery.\n")
            return BespokeJSONRecovery.recover(clean_text, expected_type)

        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            print(f"\n🛠️ [STEP 3] JSON Decode Error ({e}). Attempting syntax repair...\n")
            
            # STEP 3: Repair and Parse
            fixed_text = self._repair_json(json_candidate)
            try:
                return json.loads(fixed_text)
            except Exception as repair_e:
                print(f"\n❌ [ERROR] Syntax repair failed: {repair_e}. Falling back to Bespoke Recovery.\n")
                
                # STEP 4: Final Fallback via Bespoke Regex Recovery
                return BespokeJSONRecovery.recover(clean_text, expected_type)

    def _repair_json(self, json_str: str) -> str:
        """
        Fixes common LLM syntax errors.
        """
        json_str = json_str.strip()

        # Fix missing commas between key-value pairs
        json_str = re.sub(r'\"\s*\n\s*\"', '",\n"', json_str)

        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # Fix unescaped internal quotes
        json_str = re.sub(r'(?<=[a-zA-Z\s])"(?=[a-zA-Z\s])', "'", json_str)

        # Handle Python booleans/nulls that LLMs sometimes hallucinate
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        json_str = re.sub(r'\bNone\b', 'null', json_str)

        # Brute force closure
        if json_str.startswith('['):
            if not json_str.endswith(']'):
                if json_str.endswith('}'):
                    json_str += "]"
                else:
                    json_str += "}]"
        elif json_str.startswith('{'):
            if not json_str.endswith('}'): 
                json_str += "}"
            
        return json_str