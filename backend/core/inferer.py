from __future__ import annotations
from .models import JobDescription, CV, MappingPair, JobDescriptionFeature, Skill, MappingPair, MatchingMeta, MatchCandidate, LineageItem
from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np
import os
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

log = logging.getLogger(__name__)
import time
import json

# Try to import Llama, handle if missing
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

log = logging.getLogger(__name__)

class JobDescriptionParser:
    def __init__(self, model_path: str, context_size: int = 8192, n_threads: int = 4):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Please install it to use the parser.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        log.info(f"⏳ Loading Llama model from {model_path}...")
        t0 = time.time()
        
        # Initialize the Model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_threads=n_threads,
            n_gpu_layers=-1, # CPU only as requested
            verbose=True   # Reduce logs in production
        )
        
        log.info(f"✅ Model Loaded in {time.time() - t0:.2f} seconds!")

   
    # ==========================================
    #  1. SKILLS SCOUT AGENT (Definition-Based)
    # ==========================================
    PROMPT_SKILLS = """You are a Specialized Skills Extraction Engine.
    
### OBJECTIVE:
Scan the provided Job Description text and identify every technical tool and interpersonal trait mentioned.

### DEFINITIONS (What to look for):
1.  **hard_skill**: Look for proper nouns representing software, programming languages, hardware, platforms, frameworks, or specific methodologies. (e.g., specific names of tools).
2.  **soft_skill**: Look for adjectives or phrases describing personal character, work style, or interpersonal abilities.

### EXTRACTION LOGIC:
* Scan the text sentence by sentence.
* If a sentence mentions a tool (e.g., "Must know X"), extract "X" as a `hard_skill`.
* If a sentence lists multiple skills (e.g., "X, Y, and Z"), separate them into distinct items.
* **Atomisation:** If a responsibility says "Manage data using X", ignore the "Manage data" part and strictly extract "X".
* **Preferred Flag:** If the skill appears in a section titled "Nice to Have", "Bonus", or "Preferred", set the boolean flag `preferred` to true. Otherwise false.

### OUTPUT FORMAT (Strict JSON):
Return a single JSON object with a key "features" containing a list of objects.
Each object must have:
- "type": either "hard_skill" or "soft_skill"
- "description": the extracted string (verbatim from text)
- "preferred": boolean

NO MARKDOWN. NO PREAMBLE. RAW JSON ONLY.
"""

    # ==========================================
    #  2. RECRUITER AGENT (Definition-Based)
    # ==========================================
    PROMPT_REQUIREMENTS = """You are a Recruitment Criteria Analyzer.

### OBJECTIVE:
Analyze the Job Description to extract the mandatory barriers to entry and educational background.

### DEFINITIONS (What to look for):
1.  **experience**: Look for sentences containing NUMBERS relating to time or history. Look for phrases like "years of experience", "track record", "history of", or "previous work in".
2.  **qualification**: Look for proper nouns indicating certification bodies, academic degrees, or government licenses.
3.  **requirement**: Look for logical constraints, citizenship status, travel requirements, shift patterns, or physical requirements.

### EXTRACTION LOGIC:
* **differentiation**:
    * If it mentions a duration (e.g., "3 years"), it is `experience`.
    * If it mentions a paper credential (e.g., "MBA"), it is `qualification`.
    * If it mentions a condition of employment (e.g., "Must be US Citizen"), it is `requirement`.
* **Verbatim:** Copy the requirement text exactly as written.
* **Preferred Flag:** Check if the section header implies it is optional/preferred.

### OUTPUT FORMAT (Strict JSON):
Return a single JSON object with a key "features" containing a list of objects.
Each object must have:
- "type": "experience", "qualification", or "requirement"
- "description": the extracted string
- "preferred": boolean

NO MARKDOWN. NO PREAMBLE. RAW JSON ONLY.
"""

    # ==========================================
    #  3. CULTURE ANALYST AGENT (Definition-Based)
    # ==========================================
    PROMPT_EMPLOYER = """You are a Corporate Identity Analyst.

### OBJECTIVE:
Extract statements that define the company's identity, values, and working environment.

### DEFINITIONS (What to look for):
1.  **employer_mission**: Look for the "About Us" section. Look for statements starting with "We are", "Our mission", "We believe", or "Our goal". This describes *what* the company does and *why*.
2.  **employer_culture**: Look for descriptors of the internal environment. Look for words like "inclusive", "fast-paced", "remote-first", "collaborative", or "family-oriented".

### EXTRACTION LOGIC:
* Extract full, complete sentences.
* Do not extract generic responsibilities here. Only extract text about the *Company* itself.

### OUTPUT FORMAT (Strict JSON):
Return a single JSON object with a key "features" containing a list of objects.
Each object must have:
- "type": "employer_mission" or "employer_culture"
- "description": the extracted string (verbatim)
- "preferred": false

NO MARKDOWN. NO PREAMBLE. RAW JSON ONLY.
"""

    # ==========================================
    #  4. JOB ANALYST AGENT (Definition-Based)
    # ==========================================
    PROMPT_RESPONSIBILITIES = """You are a Job Duty Auditor.

### OBJECTIVE:
Extract every single actionable task listed in the job description.

### DEFINITIONS (What to look for):
1.  **responsibility**: Look for bullet points starting with action verbs (e.g., Manage, Design, Build, Coordinate, Assist). Look in sections titled "Responsibilities", "Duties", "What you will do", or "Role Overview".

### EXTRACTION LOGIC (Anti-Laziness):
* **Exhaustive Extraction:** You must extract EVERY bullet point found in the duties section.
* **Verbatim Protocol:** Copy the text EXACTLY as it appears.
* **Prohibited:** Do not summarize. Do not shorten. **Do not use ellipses (...)**. If a sentence is long, capture the whole string.

### OUTPUT FORMAT (Strict JSON):
Return a single JSON object with a key "features" containing a list of objects.
Each object must have:
- "type": "responsibility"
- "description": the extracted string (verbatim)
- "preferred": false

NO MARKDOWN. NO PREAMBLE. RAW JSON ONLY.
"""

    # ==========================================
    #  5. HR GENERALIST (METADATA) AGENT (Definition-Based)
    # ==========================================
    PROMPT_CORE = """You are a Compensation and Benefits Specialist.

### OBJECTIVE:
Extract the core metadata of the job listing and the benefits package.

### DEFINITIONS (What to look for):
1.  **title**: The name of the role. Usually at the very top of the text.
2.  **company**: The name of the organization hiring. Look at the top, bottom, or in copyright footers.
3.  **location**: The physical city, state, or "Remote"/"Hybrid" status.
4.  **salary_range**: Specific monetary figures (e.g., $50k, £30/hr). If not found, return null.
5.  **benefit**: Look for perks listed in "Benefits" or "What we offer" (e.g., 401k, PTO, Health, Snacks).
6.  **role_value**: High-level impact statements (e.g., "You will lead the expansion").

### EXTRACTION LOGIC:
* For metadata (title, company, location, salary), extract the most specific string found.
* For benefits, extract each item as a separate feature object.

### OUTPUT FORMAT (Strict JSON):
Return a single JSON object with keys: "title", "company", "location", "salary_range", and "features".
"features" must be a list of objects with:
- "type": "benefit" or "role_value"
- "description": the extracted string
- "preferred": boolean

NO MARKDOWN. NO PREAMBLE. RAW JSON ONLY.
"""

    
    def _repair_json(self, json_str: str) -> str:
        """Attempts to fix common JSON truncation errors."""
        json_str = json_str.strip()
        
        # 1. Close unclosed string quotes
        # Count quotes. If odd, we are inside a string.
        if json_str.count('"') % 2 != 0:
            json_str += '"'
            
        # 2. Close unclosed feature objects
        if json_str.endswith(","):
            json_str = json_str[:-1] # Remove trailing comma
        if json_str.endswith("}") and not json_str.endswith("]}") and "features" in json_str:
            # We might have closed an object but not the list
            pass 
            
        # 3. Bruteforce closure
        # If it doesn't end with '}', try adding closures until it parses or we give up
        closers = ["}", "]}", ""]
        
        for c in closers:
            try:
                candidate = json_str + c
                json.loads(candidate)
                return candidate
            except:
                continue
                
        # Fallback: Just append "]}" and hope
        if not json_str.endswith("}"):
             return json_str + "]}"
             
        return json_str


    def _run_inference(self, prompt_template: str, job_text: str, max_tokens: int = 1024) -> dict:
        full_prompt = f"""<|start_header_id|>system<|end_header_id|>

{prompt_template}<|eot_id|><|start_header_id|>user<|end_header_id|>

### JOB DESCRIPTION:
{job_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        output = self.llm(
            full_prompt,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
            temperature=0.0,
            echo=False,
            repeat_penalty=1.05,
            top_p=0.95
        )
        
        result_text = output['choices'][0]['text'].strip()
        
        # Cleanup Markdown
        if "```" in result_text:
            result_text = result_text.split("```")[-1]
            if result_text.startswith("json"): result_text = result_text[4:]
            result_text = result_text.split("```")[0]
            
        result_text = result_text.strip()
        
        # Find JSON start
        start_idx = result_text.find("{")
        if start_idx != -1:
            result_text = result_text[start_idx:]

        print(f"🔍 Raw LLM Output: {result_text}")  # Log last 100 chars for debugging
        
        # Attempt Repair
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            log.warning(f"⚠️ JSON Decode Error. Attempting repair on: {result_text[-50:]}...")
            fixed_text = self._repair_json(result_text)
            try:
                return json.loads(fixed_text)
            except:
                log.error("❌ Failed to repair JSON.")
                return {}
    


    def slow_parse(self, job_text: str) -> dict:
        """
        Orchestrates the 5 specialized agents.
        """
        log.info("🐢 Starting Slow Parse (5-Step Process)...")
        start_time = time.time()

        # 1. Run the Agents
        skills_data = self._run_inference(self.PROMPT_SKILLS, job_text)
        reqs_data = self._run_inference(self.PROMPT_REQUIREMENTS, job_text)
        empl_data = self._run_inference(self.PROMPT_EMPLOYER, job_text)
        resp_data = self._run_inference(self.PROMPT_RESPONSIBILITIES, job_text, max_tokens=2048)
        core_data = self._run_inference(self.PROMPT_CORE, job_text)

        # 2. Merge Results
        final_result = {
            "title": core_data.get("title"),
            "company": core_data.get("company"),
            "location": core_data.get("location"),
            "salary_range": core_data.get("salary_range"),
            "features": [],
            "_meta": {}
        }

        # Combine all features lists safely
        all_features = (
            (skills_data.get("features") or []) +
            (reqs_data.get("features") or []) +
            (empl_data.get("features") or []) +
            (resp_data.get("features") or []) +
            (core_data.get("features") or [])
        )

        # 3. Safe Deduplication
        seen = set()
        unique_features = []
        
        for f in all_features:
            if not isinstance(f, dict):
                continue
            
            raw_desc = f.get("description")
            if not raw_desc: 
                continue 
                
            desc = str(raw_desc).strip()
            sig = (f.get("type"), desc.lower())
            
            if sig not in seen:
                seen.add(sig)
                f["description"] = desc 
                unique_features.append(f)

        final_result["features"] = unique_features
        final_result["_meta"]["generation_time_sec"] = round(time.time() - start_time, 2)
        final_result["_meta"]["method"] = "slow_parse_v1"
        
        log.info(f"✅ Slow Parse Complete in {final_result['_meta']['generation_time_sec']}s")
        return final_result
    

    
   # ==========================================
    #  AGENT 1: THE CANDIDATE PROFILE (WHO)
    # ==========================================
    PROMPT_CANDIDATE = """You are a Technical Resume Screener.

### OBJECTIVE:
Extract CANDIDATE ATTRIBUTES (what they know/have). You must ignore Daily Duties (what they do).

### SEMANTIC DEFINITIONS:
1. "hard_skill":
   - MATCH: Specific Proper Nouns representing Tools, Technologies, Standards, or Methodologies.
   - CATEGORIES: Software names, Hardware/Machinery names, Regulatory Frameworks, Technical Protocols.
   - REJECT: General business processes, administrative tasks, or sentences describing a duty (e.g., "Managing the team").

2. "soft_skill":
   - MATCH: Adjectives or short phrases defining personal character or work style.
   - CONSTRAINT: Max 3 words.

3. "certification":
   - MATCH: **Paper Credentials ONLY**. Academic Degrees, Professional Certifications, Government Licenses, Security Clearances.
   - REJECT: Phrases like "Knowledge of", "Experience in", or "Ability to".

### OUTPUT SCHEMA (STRICT JSON):
You must output a single, raw JSON object matching this exact structure:
{
  "features": [
    {
      "type": "string (Must be one of: hard_skill, soft_skill, certification)",
      "description": "string (The extracted text verbatim)",
    }
  ]
}
DO NOT RETURN MARKDOWN. DO NOT USE ```json BLOCKS. JUST THE RAW JSON.
"""

    # ==========================================
    #  AGENT 2: THE JOB EXECUTION (WHAT)
    # ==========================================
    PROMPT_JOB = """You are a Job Analyst.

### OBJECTIVE:
Extract DAILY DUTIES and LOGISTICAL CONSTRAINTS.

### SEMANTIC DEFINITIONS:
1. "responsibility":
   - DEFINITION: **Actions**. Things the employee DOES on a daily basis.
   - SIGNAL: Bullet points or sentences starting with **Action Verbs**.
   - RULE: If it describes a task to be performed, it is a Responsibility.

2. "requirement":
   - DEFINITION: **All Mandatory Constraints** (Logistics, Legal, Status, OR History).
   - LIKE:
     a) **Logistics/Legal:** Citizenship, Visa, Criminal checks, Shifts, Travel %, Location.
     b) **History/Experience:** Years of experience, Proven track record, Previous tenure.
   - **CRITICAL INSTRUCTION:** If the requirement is historical (Time/Experience based), you MUST append " (experience)" to the end of the string.
   - REJECT: Any bullet point starting with an Action Verb (Move those to Responsibility).

### OUTPUT SCHEMA (STRICT JSON):
You must output a single, raw JSON object matching this exact structure:
{
  "features": [
    {
      "type": "string (Must be one of: responsibility, requirement)",
      "description": "string (The extracted text verbatim)",
    }
  ]
}
DO NOT RETURN MARKDOWN. DO NOT USE ```json BLOCKS. JUST THE RAW JSON.
"""

    # ==========================================
    #  AGENT 3: THE VALUE PROPOSITION (WHY/WHERE)
    # ==========================================
    PROMPT_VALUE = """You are a Compensation Analyst.

### OBJECTIVE:
Extract Rewards and Metadata.

### METADATA:
- "title": Specific role name.
- "company": Organization name.
- "location": City, State, or Remote status.
- "salary_range":
   - DEFINITION: The specific amount of money the job pays.
   - LOOK FOR: Currency symbols (£, $, €) and numbers.
   - MATCH: ** You may ONLY extract items that appear listed under headers like "Salary", "Compensation", "Pay", "Remuneration" or immediately following labels like "Salary:".
   - EXAMPLES: "£30,000 - £40,000", "$50k per year", "£12.50/hr", "€600 per day".

### FEATURE DEFINITIONS:

1. "employer_culture":
   - MATCH: Adjectives or phrases or slogans describing the company culture or vibe, ETHOS, VALUES, ETHICS
2. "employer_mission":
   - MATCH: High-level purpose statements describing what the company does and why.
3. "perk":
   - LIKE: Bonus, Pension, Health coverage, Time off, Lifestyle perks (Gym, Food), Equipment, Discounts.
   - MATCH: ** You may ONLY extract items that appear listed under headers like "Benefits", "Perks", "Rewards", "What we offer", "from us?", "offers", "what you get", "What you can expect from us" or immediately following labels like "Benefit:".


### OUTPUT SCHEMA (STRICT JSON):
You must output a single, raw JSON object matching this exact structure:
{
  "title": "string or null",
  "company": "string or null",
  "location": "string or null",
  "salary_range": "string or null",
  "features": [
    {
      "type": "string (Must be one of: employer_culture, employer_mission, perk)",
      "description": "string (The extracted text)",
    }
  ]
}
DO NOT RETURN MARKDOWN. DO NOT USE ```json BLOCKS. JUST THE RAW JSON.
"""

    def medium_parse(self, job_text: str) -> dict:
        log.info("🐇 Starting Medium Parse (3-Agent Architecture v34)...")
        start_time = time.time()

        # 1. Run 3 Agents Sequentially
        candidate_data = self._run_inference(self.PROMPT_CANDIDATE, job_text, max_tokens=2500)
        job_data = self._run_inference(self.PROMPT_JOB, job_text, max_tokens=2500)
        value_data = self._run_inference(self.PROMPT_VALUE, job_text, max_tokens=1500)

        # 2. Initialize Final Result
        final_result = {
            "title": value_data.get("title"),
            "company": value_data.get("company"),
            "location": value_data.get("location"),
            "salary_range": value_data.get("salary_range"),
            "features": [],
            "_meta": {}
        }

        # 3. Combine Features
        all_features = (
            (candidate_data.get("features") or []) +
            (job_data.get("features") or []) +
            (value_data.get("features") or [])
        )

        # 4. Dedup, Clean, and Validate
        seen = set()
        unique_features = []
        
        VALID_TYPES = [
            "responsibility", "hard_skill", "soft_skill", "experience",
            "qualification", "requirement", "nice_to_have", 
            "employer_mission", "employer_culture", "role_value", 
            "benefit", "other"
        ]

        for f in all_features:
            if not isinstance(f, dict): continue
            
            # Safe String Handling
            raw_val = f.get("description")
            if raw_val is None: continue 
            clean_desc = str(raw_val).strip()
            
            if not clean_desc: continue
            
            feat_type = f.get("type")

            # --- MAPPING LOGIC ---
            if feat_type == "certification":
                feat_type = "qualification"
            if feat_type == "perk":
                feat_type = "benefit"
            # ---------------------

            # Validate Type
            if feat_type not in VALID_TYPES:
                # If Agent 1 returned 'requirement' (with experience suffix), it's valid.
                if feat_type == "requirement": 
                    pass 
                else: 
                    continue
            
            # Anti-Hallucination Checks
            if clean_desc.lower().startswith("we are looking"): continue
            if clean_desc == final_result.get("location"): continue
            
            sig = (feat_type, clean_desc.lower())
            
            if sig not in seen:
                seen.add(sig)
                f["type"] = feat_type
                f["description"] = clean_desc
                f["preferred"] = bool(f.get("preferred", False))
                unique_features.append(f)

        final_result["features"] = unique_features
        final_result["_meta"]["generation_time_sec"] = round(time.time() - start_time, 2)
        final_result["_meta"]["method"] = "medium_parse_v34_refined"
        
        log.info(f"✅ Medium Parse Complete in {final_result['_meta']['generation_time_sec']}s")
        return final_result
    
    def parse(self, job_text: str) -> dict:
        return self.medium_parse(job_text)
        # return self.slow_parse(job_text)
    

    



#     def parse(self, job_text: str) -> dict:
#         """
#         Runs the Llama model on the job text and returns a Python dictionary.
#         """
#         if not job_text or not job_text.strip():
#             return {"error": "Empty job text provided"}

#         # Llama-3 specific formatting
#         # NEW (Fix)
#         full_prompt = f"""<|start_header_id|>system<|end_header_id|>

# {self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

# ### JOB DESCRIPTION:
# {job_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

#         log.info("🤖 Analyzing Job Description...")
#         start_gen = time.time()
        
#         output = self.llm(
#             full_prompt,
#             max_tokens=4096,      # Increased from 2048 to ensure full JSON fits
#             stop=["<|eot_id|>"],
#             temperature=0.1, 
#             echo=False,
#             repeat_penalty=1.05,  # <--- NEW: Penalizes repeating lines
#             top_p=0.9,             # <--- NEW: Helps break out of loops
#             top_k=50
#         )
        
#         generation_time = time.time() - start_gen
#         result_text = output['choices'][0]['text'].strip()
        
#         # Clean up potential Markdown formatting if the model adds it
#         # Clean up potential Markdown formatting
#         if result_text.startswith("```json"):
#             result_text = result_text.replace("```json", "").replace("```", "")
        
#         result_text = result_text.strip()

#         # --- REPAIR LOGIC START ---
#         # 1. Fix missing closing brace (The error you are seeing)
#         if result_text.endswith("]") and not result_text.endswith("}"):
#             log.warning("⚠️ Llama forgot the closing brace. Auto-repairing...")
#             result_text += "}"
        
#         # 2. Fix missing list closure (Common fallback)
#         if not result_text.endswith("}") and not result_text.endswith("]"):
#              # If it just stopped in the middle of nowhere, try to close everything
#              result_text += "]}"
#         # --- REPAIR LOGIC END ---

#         try:
#             parsed_json = json.loads(result_text)

#             # --- CLEANUP LOGIC ---
#             # Your model is outputting "description": null, which might break things later.
#             # Let's remove those useless features.
#             if "features" in parsed_json and isinstance(parsed_json["features"], list):
#                 parsed_json["features"] = [
#                     f for f in parsed_json["features"] 
#                     if f.get("description") is not None and f.get("description") != "None"
#                 ]

#             parsed_json["_meta"] = {"generation_time_sec": round(generation_time, 2)}
#             return parsed_json

#         except json.JSONDecodeError:
#             log.error(f"Failed to decode JSON. Raw output: {result_text}")
#             return {"error": "Failed to generate valid JSON", "raw_output": result_text}
        


# =====================================================
# Utilities
# =====================================================
def normalize_text(text: str) -> str:
    """Simple normalization: lowercase, strip punctuation, squeeze spaces."""
    if not text: return ""
    # Keep alphanumeric, spaces, and essential punctuation (+ for C++, # for C#)
    text = re.sub(r"[^a-zA-Z0-9\s\.\-\+\#]", " ", text) 
    return " ".join(text.lower().split())

def intelligent_segmentation(text: str, nlp_model=None) -> List[str]:
    """
    Robust segmentation that handles bullet points AND sentences.
    1. Splits by newlines/bullets first.
    2. Then splits by sentences using spaCy (if available).
    """
    if not text: return []
    
    # Step 1: Structural Split (Bullets/Newlines)
    # Matches: \n, \r, \t, "• ", "· ", "- ", "* "
    raw_chunks = re.split(r'\n|\r|\t|•\s|·\s|\-\s|\*\s', text)
    
    final_segments = []
    
    for chunk in raw_chunks:
        clean_chunk = chunk.strip()
        # Skip empty/tiny fragments
        if len(clean_chunk) < 5: continue 
        
        # Step 2: Linguistic Split (spaCy)
        # If the chunk is long (likely a paragraph), break it down.
        if nlp_model and len(clean_chunk) > 80:
            doc = nlp_model(clean_chunk)
            for sent in doc.sents:
                if len(sent.text) > 10:
                    final_segments.append(sent.text.strip())
        else:
            final_segments.append(clean_chunk)
            
    return final_segments



# =====================================================
# 1. Custom Model Definition & Embedder
# =====================================================

class DebertaMultiLabel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=False)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits

class JobSegmentEmbedder:
    def __init__(self, checkpoint_path: str, model_name: str = "microsoft/deberta-v3-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        log.info(f"Loading custom Job Segment Model from {checkpoint_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = DebertaMultiLabel(model_name, num_classes=12)
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get('model_state', checkpoint)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
            else:
                log.error(f"Checkpoint not found. Semantic matching disabled.")
                self.model = None
        except Exception as e:
            log.error(f"Failed to load custom model: {e}")
            self.model = None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.model or not texts: return None
        embeddings = []
        batch_size = 16 
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
                input_ids = encoded_input['input_ids'].to(self.device)
                attention_mask = encoded_input['attention_mask'].to(self.device)
                with torch.no_grad():
                    outputs = self.model.backbone(input_ids=input_ids, attention_mask=attention_mask)
                    cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_vectors)
            except Exception:
                return None
        return np.vstack(embeddings) if embeddings else None

# =====================================================
# 2. Atomic Evidence Model (Structured Lineage)
# =====================================================

class CVEvidence:
    """
    Represents a Unique Semantic Atom.
    Stores STRUCTURED LINEAGE (List of Dicts) for direct Pydantic conversion.
    """
    def __init__(self, text: str, item_type: str, lineage_path: List[Dict[str, str]]):
        self.text = text
        self.item_type = item_type
        
        # Occurrences store the full path to the evidence
        # path format: [{"id": "123", "type": "project", "name": "App"}, ...]
        self.occurrences: List[List[Dict[str, str]]] = []
        self.add_occurrence(lineage_path)

        self.vector_tfidf: Optional[np.ndarray] = None
        self.vector_embedding: Optional[np.ndarray] = None

    def add_occurrence(self, path: List[Dict[str, str]]):
        # Deduplicate based on the ID of the leaf node
        if not path: return
        leaf_id = path[-1]['id']
        for occ in self.occurrences:
            if occ[-1]['id'] == leaf_id:
                return
        self.occurrences.append(path)

# =====================================================
# 3. MappingInferer (Unlimited Results)
# =====================================================

class MappingInferer:
    def __init__(self):
        self.default_min_score = 0.28
        self.embedder: Optional[JobSegmentEmbedder] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.nlp: Optional[Any] = None
        self.CHECKPOINT_PATH = "backend/core/job_model/checkpoints/model_best.pt"

        self.type_matrix = {
            ("hard_skill", "skill"): 0.20, ("soft_skill", "skill"): 0.15,
            ("responsibility", "achievement"): 0.15, ("hard_skill", "achievement"): 0.15,
            ("responsibility", "experience"): 0.10, ("hard_skill", "project"): 0.10,
            ("qualification", "education"): 0.15, ("benefit", "hobby"): 0.05
        }

    def load_models(self):
        log.info("Loading spaCy model...")
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                pass
        log.info("Initializing JobSegmentEmbedder...")
        self.embedder = JobSegmentEmbedder(self.CHECKPOINT_PATH)

    def _preprocess_cv(self, cv: CV) -> List[CVEvidence]:
        """
        Deconstructs CV into atoms with STRUCTURED breadcrumb chains.
        """
        atom_registry: Dict[Tuple[str, str], CVEvidence] = {}
        
        # 1. Build Context Map (ID -> List[Dict] Chain)
        context_chains = {}
        for exp in getattr(cv, "experiences", []):
            context_chains[exp.id] = [{"id": exp.id, "type": "experience", "name": f"{exp.title} at {exp.company}"}]
        for edu in getattr(cv, "education", []):
            context_chains[edu.id] = [{"id": edu.id, "type": "education", "name": f"{edu.degree}"}]
        for hobby in getattr(cv, "hobbies", []):
            context_chains[hobby.id] = [{"id": hobby.id, "type": "hobby", "name": hobby.name}]

        ach_map = {a.id: a for a in getattr(cv, "achievements", [])}
        skill_map = {s.id: s for s in getattr(cv, "skills", [])}

        def register_atom(text, item_type, path_chain):
            norm_text = normalize_text(text)
            if not norm_text: return
            key = (norm_text, item_type)
            
            if key in atom_registry:
                atom_registry[key].add_occurrence(path_chain)
            else:
                atom = CVEvidence(norm_text, item_type, path_chain)
                atom_registry[key] = atom

        def add_children(parent_obj, parent_chain):
            # Achievements
            for ach_id in getattr(parent_obj, "achievement_ids", []):
                ach = ach_map.get(ach_id)
                if ach and getattr(ach, 'text', None):
                    ach_chain = parent_chain + [{
                        "id": ach.id, 
                        "type": "achievement", 
                        "name": (ach.text[:20] + "...") 
                    }]
                    full_text = f"{ach.text} {getattr(ach, 'context', '') or ''}".strip()
                    register_atom(full_text, "achievement", ach_chain)

                    # Skills inside Achievements
                    for sk_id in getattr(ach, "skill_ids", []):
                        sk = skill_map.get(sk_id)
                        if sk:
                            sk_chain = ach_chain + [{"id": sk.id, "type": "skill", "name": sk.name}]
                            register_atom(sk.name, "skill", sk_chain)

            # Direct Skills
            for sk_id in getattr(parent_obj, "skill_ids", []):
                sk = skill_map.get(sk_id)
                if sk:
                    sk_chain = parent_chain + [{"id": sk.id, "type": "skill", "name": sk.name}]
                    register_atom(sk.name, "skill", sk_chain)

        # --- 1. Experiences ---
        for exp in getattr(cv, "experiences", []):
            base_chain = context_chains[exp.id] 
            if exp.title:
                register_atom(exp.title, "experience", base_chain) 
            
            for seg in intelligent_segmentation(getattr(exp, 'description', '') or '', self.nlp):
                seg_id = f"seg-{hash(seg)}"
                seg_chain = base_chain + [{"id": seg_id, "type": "description", "name": "Description"}]
                register_atom(seg, "experience", seg_chain)
            add_children(exp, base_chain)

        # --- 2. Projects ---
        for proj in getattr(cv, "projects", []):
            base_chain = []
            found_parent = False
            for rel_id in getattr(proj, "related_experience_ids", []) + getattr(proj, "related_education_ids", []) + getattr(proj, "related_hobby_ids", []):
                if rel_id in context_chains:
                    base_chain = context_chains[rel_id] + [{"id": proj.id, "type": "project", "name": proj.title}]
                    found_parent = True
                    break 
            
            if not found_parent:
                base_chain = [{"id": proj.id, "type": "project", "name": proj.title}]

            if proj.title:
                register_atom(proj.title, "project", base_chain)

            for seg in intelligent_segmentation(getattr(proj, 'description', '') or '', self.nlp):
                seg_id = f"seg-{hash(seg)}"
                seg_chain = base_chain + [{"id": seg_id, "type": "description", "name": "Description"}]
                register_atom(seg, "project", seg_chain)
            add_children(proj, base_chain)

        # --- 3. Education ---
        for edu in getattr(cv, "education", []):
            base_chain = context_chains[edu.id]
            core_text = f"{edu.degree} {getattr(edu, 'field', '')}".strip()
            if core_text:
                register_atom(core_text, "education", base_chain)
            add_children(edu, base_chain)

        # --- 4. Hobbies ---
        for hobby in getattr(cv, "hobbies", []):
            base_chain = context_chains[hobby.id]
            if hobby.name:
                register_atom(hobby.name, "hobby", base_chain)
            for seg in intelligent_segmentation(getattr(hobby, 'description', '') or '', self.nlp):
                seg_id = f"seg-{hash(seg)}"
                seg_chain = base_chain + [{"id": seg_id, "type": "description", "name": "Description"}]
                register_atom(seg, "hobby", seg_chain)
            add_children(hobby, base_chain)

        # --- 5. Global Skills ---
        for sk in getattr(cv, "skills", []):
            if sk.name:
                chain = [{"id": sk.id, "type": "skill", "name": sk.name}]
                register_atom(sk.name, "skill", chain)

        return list(atom_registry.values())

    def _build_vectors(self, req_texts: List[str], cv_texts: List[str]):
        corpus = req_texts + cv_texts
        self.tfidf = TfidfVectorizer(stop_words='english')
        try:
            all_tfidf = self.tfidf.fit_transform(corpus).toarray()
            req_tfidf = all_tfidf[:len(req_texts)]
            cv_tfidf = all_tfidf[len(req_texts):]
        except ValueError:
            return None, None, None, None

        req_emb, cv_emb = None, None
        if self.embedder:
            all_emb = self.embedder.encode(corpus)
            if all_emb is not None:
                req_emb = all_emb[:len(req_texts)]
                cv_emb = all_emb[len(req_texts):]
        
        return req_tfidf, req_emb, cv_tfidf, cv_emb

    def infer_mappings(self, job: JobDescription, cv: CV, min_score: Optional[float] = None, top_k: Optional[int] = None, **kwargs) -> List[MappingPair]:
        active_min_score = min_score if min_score is not None else self.default_min_score
        
        # NOTE: top_k logic has been removed to ensure ALL valid matches are returned 
        # for proper CV Generation. Frontend will handle slicing.

        if not self.embedder: self.load_models()

        job_features = [
            f for f in getattr(job, "features", []) 
            if f.description and f.type not in ['benefit', 'other']
        ]
        evidence_pool = self._preprocess_cv(cv)

        if not job_features or not evidence_pool: return []

        req_texts = [normalize_text(f.description) for f in job_features]
        cv_texts = [normalize_text(e.text) for e in evidence_pool]

        r_tf, r_emb, c_tf, c_emb = self._build_vectors(req_texts, cv_texts)
        if r_tf is None: return []

        mappings = []
        
        for i, req in enumerate(job_features):
            
            # 1. Collect All Valid Matches (Quality Gate)
            candidates = []
            for j, ev in enumerate(evidence_pool):
                tfidf_sim = float(cosine_similarity([r_tf[i]], [c_tf[j]])[0][0])
                emb_sim = 0.0
                if r_emb is not None and c_emb is not None:
                    emb_sim = float(cosine_similarity([r_emb[i]], [c_emb[j]])[0][0])
                
                if tfidf_sim > 0.7: base_score = tfidf_sim
                else: base_score = (0.3 * tfidf_sim) + (0.7 * emb_sim) if emb_sim else tfidf_sim

                if req.type in ["hard_skill", "qualification", "tool"]:
                    if tfidf_sim < 0.1: base_score = 0.0
                elif tfidf_sim < 0.05: base_score = base_score * 0.4 

                type_bonus = 0.0
                if base_score > 0.35:
                     type_bonus = self.type_matrix.get((req.type, ev.item_type), 0.0)
                
                final_score = min(base_score + type_bonus, 1.0)
                
                if final_score >= active_min_score:
                    candidates.append({"score": final_score, "evidence": ev})

            # 2. Group by Root Parent (Deduplication)
            # We group candidates by the ID of the root item in their lineage chain
            parent_groups = defaultdict(list)
            for cand in candidates:
                root_id = cand['evidence'].occurrences[0][0]['id']
                parent_groups[root_id].append(cand)

            # 3. Create MappingPairs for Groups
            group_results = []
            
            for pid, group_hits in parent_groups.items():
                group_hits.sort(key=lambda x: x['score'], reverse=True)
                
                best_hit = group_hits[0]
                best_ev = best_hit['evidence']
                best_path = best_ev.occurrences[0] # List[Dict]

                # --- SMART NOTE GENERATION WITH STRUCTURED DATA ---
                
                # A. Convert all lineages to strings for text processing
                raw_locations_str = []
                for occ in best_ev.occurrences:
                     path_str = " > ".join([f"{step['type'].title()}: {step['name']}" for step in occ])
                     raw_locations_str.append(path_str)
                
                # B. Apply Text Filtering Logic (Remove redundant prefixes)
                unique_locs = sorted(list(set(raw_locations_str)), key=len, reverse=True)
                final_locs = []
                for loc in unique_locs:
                    is_redundant = any(loc in accepted for accepted in final_locs)
                    if not is_redundant:
                        final_locs.append(loc)

                primary_loc_str = final_locs[0]
                others = final_locs[1:]
                
                excerpt = best_ev.text[:72] + "..." if len(best_ev.text) > 75 else best_ev.text
                note = f"Excerpt: \"{excerpt}\" ({primary_loc_str})"
                
                if others:
                    extras = "; ".join(others[:2])
                    if len(others) > 2: extras += f"; +{len(others)-2} more"
                    note += f" [Also found in: {extras}]"
                
                confidence = int(best_hit['score'] * 100)
                note += f" [Confidence: {confidence}%]"
                if best_hit['score'] > 0.85: note += " - Strong Match"
                
                # -----------------------------------------------------------

                # A. Build Best Match Candidate (Pydantic)
                best_candidate = MatchCandidate(
                    segment_text=best_ev.text,
                    segment_type=best_ev.item_type,
                    score=best_hit['score'],
                    lineage=[LineageItem(**step) for step in best_path]
                )

                # B. Build Supporting Candidates (Limit to top 2 for display sanity, but keep parent group integrity)
                supporting = []
                seen_texts = {best_ev.text}
                for hit in group_hits[1:]:
                    ev = hit['evidence']
                    if ev.text not in seen_texts and len(supporting) < 2:
                        path = ev.occurrences[0]
                        supporting.append(MatchCandidate(
                            segment_text=ev.text,
                            segment_type=ev.item_type,
                            score=hit['score'],
                            lineage=[LineageItem(**step) for step in path]
                        ))
                        seen_texts.add(ev.text)

                # C. Build Meta Container
                meta_obj = MatchingMeta(
                    best_match=best_candidate,
                    supporting_matches=supporting,
                    summary_note=note # Detailed smart note
                )

                # D. Create Mapping Pair
                root_container = best_path[0]
                
                pair = MappingPair(
                    feature_id=req.id,
                    feature_text=req.description,
                    context_item_id=root_container['id'],
                    context_item_type=root_container['type'],
                    context_item_text=root_container['name'],
                    strength=best_hit['score'],
                    meta=meta_obj,
                    annotation=meta_obj.summary_note # Backward compatibility
                )
                
                group_results.append(pair)

            # 4. Final Processing: Sort and Return ALL
            # Sort by strength so the generator sees best options first
            group_results.sort(key=lambda x: x.strength, reverse=True)
            mappings.extend(group_results)

        return mappings