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
    def __init__(self, model_path: str, context_size: int = 8192, n_threads: int = 4, machine_type: str = "mac"):
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Please install it to use the parser.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        log.info(f"⏳ Loading Llama model from {model_path}...")
        t0 = time.time()
        
        
        # Initialize the Model
        if machine_type == "mac":
            self.llm = Llama(
                model_path=model_path,
                n_ctx=context_size,
                n_threads=n_threads,
                n_gpu_layers=-1, # CPU only as requested
                verbose=True,   # Reduce logs in production
                flash_attn=True,
            )
        else:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_batch=1024,        # SPEEDUP: Processes your long "Master Prompt" in larger chunks.
                n_threads=6,
                n_gpu_layers=0, # CPU only as requested
                flash_attn=True,
                verbose=True,   # Reduce logs in production
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
        """Attempts to fix common JSON errors: truncation AND missing commas."""
        json_str = json_str.strip()

        # 1. Fix missing commas between key-value pairs
        # Look for: "value" "next_key": 
        # We insert a comma if it's missing between a double quote and the next key
        json_str = re.sub(r'\"\s*\n\s*\"', '",\n"', json_str)
        
        # 2. Close unclosed string quotes
        if json_str.count('"') % 2 != 0:
            json_str += '"'
            
        # 3. Remove trailing commas (e.g., "item", } -> "item" })
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

        # 4. Bruteforce closure (Truncation handling)
        # If it doesn't end with '}', try adding closures
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
    
    # ==========================================
    #  MASTER AGENT (COMBINED SINGLE-PASS)
    # ==========================================
    PROMPT_MASTER = """You are a Lead HR Analyst.

### OBJECTIVE:
Analyze the Job Description and extract a complete Job Profile. You must extract Metadata, Candidate Attributes (Who), Job Execution Details (What), and Value Proposition (Why).

### SECTION 1: METADATA
Extract the following specific fields. If not found, set to null.
- "title": Specific role name.
- "company": Organization name.
- "location": City, State, or Remote status.
- "salary_range": Specific monetary figures (e.g., "$50k - $80k", "£15/hr"). IGNORE generic "competitive salary" text.

### SECTION 2: SEMANTIC FEATURE DEFINITIONS
Extract specific text segments into a "features" list based on these types:

#### GROUP A: CANDIDATE ATTRIBUTES (The "Who")
1. "hard_skill": 
   - MATCH: Proper Nouns for Tools, Software, Frameworks, Methodologies, or Tech Stacks.
   - REJECT: General business processes, administrative tasks, or duties.
2. "soft_skill": 
   - MATCH: Adjectives defining character/work style (Max 3 words).
3. "certification": 
   - MATCH: Paper credentials ONLY (Degrees, Licenses, Security Clearances).

#### GROUP B: JOB EXECUTION (The "What")
4. "responsibility": 
   - MATCH: Actions the employee DOES daily. Look for bullet points starting with Action Verbs.
5. "requirement": 
   - MATCH: Mandatory constraints (Logistics, Legal status, History).
   - CRITICAL RULE: If the requirement is historical (e.g., "3 years experience"), you MUST append " (experience)" to the string.

#### GROUP C: VALUE PROPOSITION (The "Why")
6. "employer_culture": 
   - MATCH: Adjectives/slogans describing vibe, ethos, or values.
7. "employer_mission": 
   - MATCH: High-level purpose statements (What the company does and why).
8. "perk": 
   - MATCH: Benefits, bonuses, insurance, time off, or lifestyle rewards.

### OUTPUT SCHEMA (STRICT JSON):
Return a single JSON object with no markdown formatting:
{
  "title": "string or null",
  "company": "string or null",
  "location": "string or null",
  "salary_range": "string or null",
  "features": [
    {
      "type": "string (Must be one of: hard_skill, soft_skill, certification, responsibility, requirement, employer_culture, employer_mission, perk)",
      "description": "string (The extracted text verbatim; seperate with semi-colon if multiple found)"
    }
  ]
}
DO NOT RETURN MARKDOWN. DO NOT USE ```json BLOCKS. JUST THE RAW JSON.
"""
    def fast_parse(self, job_text: str) -> dict:
        """
        Runs the entire extraction in a single prompt pass (approx 3x faster).
        Splits bundled items (;) into atomic features and fixes reference bugs.
        """
        log.info("⚡ Starting Fast Parse (Master Agent v1)...")
        start_time = time.time()

        # 1. Run Inference (Max tokens increased to 4096 to fit the larger JSON)
        data = self._run_inference(self.PROMPT_MASTER, job_text, max_tokens=4096)

        # 2. Initialize Final Result Structure
        final_result = {
            "title": data.get("title"),
            "company": data.get("company"),
            "location": data.get("location"),
            "salary_range": data.get("salary_range"),
            "features": [],
            "_meta": {}
        }

        # 3. Process Features (Dedup, Map, and Validate)
        raw_features = data.get("features", [])
        seen = set()
        unique_features = []

        # Define allowed types (matching your frontend/database schema)
        VALID_TYPES = [
            "responsibility", "hard_skill", "soft_skill", "experience",
            "qualification", "requirement", "nice_to_have", 
            "employer_mission", "employer_culture", "role_value", 
            "benefit", "other"
        ]

        for f in raw_features:
            if not isinstance(f, dict): continue
            
            # Safe String Handling
            raw_val = f.get("description")
            if raw_val is None: continue 
            clean_desc_full = str(raw_val).strip()
            if not clean_desc_full: continue
            
            feat_type = f.get("type")

            # --- MAPPING LOGIC ---
            if feat_type == "certification":
                feat_type = "qualification"
            if feat_type == "perk":
                feat_type = "benefit"
            # ---------------------

            # Validate Type (Allow 'requirement' as it is a core extraction type here)
            if feat_type not in VALID_TYPES and feat_type != "requirement":
                continue

            # --- UNBUNDLING LOGIC (Fixing the loop bug) ---
            # We split by semicolon to unpack the bundled string
            for raw_item in clean_desc_full.split(';'):
                clean_desc = raw_item.strip() # Remove leading/trailing spaces
                
                # Skip empty items or tiny artifacts
                if not clean_desc or len(clean_desc) < 2: continue

                # Anti-Hallucination Checks
                if clean_desc.lower().startswith("we are looking"): continue
                if clean_desc == final_result.get("location"): continue
                
                # Deduplication Signature
                sig = (feat_type, clean_desc.lower())
                
                if sig not in seen:
                    seen.add(sig)
                    
                    # -------------------------------------------------------
                    # CRITICAL FIX: Create a FRESH dictionary object here.
                    # Do NOT modify or append 'f', or it will overwrite previous items.
                    # -------------------------------------------------------
                    new_feature = {
                        "type": feat_type,
                        "description": clean_desc,
                        "preferred": bool(f.get("preferred", False))
                    }
                    unique_features.append(new_feature)

        final_result["features"] = unique_features
        final_result["_meta"]["generation_time_sec"] = round(time.time() - start_time, 2)
        final_result["_meta"]["method"] = "fast_parse_master_v1"
        
        log.info(f"✅ Fast Parse Complete in {final_result['_meta']['generation_time_sec']}s")
        return final_result


    def parse(self, job_text: str) -> dict:
        # return self.medium_parse(job_text)
        # return self.slow_parse(job_text)
        return self.fast_parse(job_text)
    

    # CV PARSER
    def _run_inference_cv(self, prompt_template: str, doc_text: str, max_tokens: int = 1024, doc_label: str = "JOB DESCRIPTION") -> dict:
        """
        Runs Llama 3 inference with robust Markdown cleaning.
        BACKWARD COMPATIBLE: doc_label defaults to "JOB DESCRIPTION".
        """
        # We explicitly label the user input so the model understands context (CV vs Job)
        full_prompt = f"""<|start_header_id|>system<|end_header_id|>

{prompt_template}<|eot_id|><|start_header_id|>user<|end_header_id|>

### {doc_label}:
{doc_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        output = self.llm(
            full_prompt,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
            temperature=0.1, 
            echo=False,
            repeat_penalty=1.05,
            top_p=0.95
        )
        
        result_text = output['choices'][0]['text'].strip()
        
        # =====================================================
        # ROBUST MARKDOWN CLEANUP (Fixes the "Last Element" Bug)
        # =====================================================
        # If the model wraps output in ```json ... ```, we extract just the code block.
        if "```" in result_text:
            parts = result_text.split("```")
            found_json = False
            for p in parts:
                clean_p = p.strip()
                # Remove language tag if present (e.g. "json")
                if clean_p.lower().startswith("json"):
                    clean_p = clean_p[4:].strip()
                # If it looks like a JSON object, take it
                if clean_p.startswith("{"):
                    result_text = clean_p
                    found_json = True
                    break
            
            # Fallback: if we didn't find a clean block, try the last part
            if not found_json:
                 # This handles cases where the model puts text AFTER the code block
                 for p in reversed(parts):
                     if p.strip().startswith("{"):
                         result_text = p.strip()
                         break

        result_text = result_text.strip()
        
        # Defensive: Ignore conversational preambles (e.g. "Here is the JSON:")
        start_idx = result_text.find("{")
        if start_idx != -1:
            result_text = result_text[start_idx:]

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            log.warning(f"⚠️ JSON Decode Error. Attempting repair...")
            fixed_text = self._repair_json(result_text)
            try:
                return json.loads(fixed_text)
            except:
                log.error("❌ Failed to repair JSON.")
                return {}
    
    
    # ==========================================
    #  CV AGENT 1: IDENTITY & METADATA
    # ==========================================
    PROMPT_CV_IDENTITY = """You are a Lead CV Analyst.

### OBJECTIVE:
Extract candidate metadata and a global skill set.

### SECTION 1: METADATA RULES
- "first_name": Extract the first name.
- "last_name": Extract the last name.
- "email": MATCH valid email addresses.
- "phone": MATCH phone numbers. If none, return null.
- "linkedin": MATCH LinkedIn profile URLs. If none, return null.
- "location": MATCH City/Country.
- "summary": MATCH the professional summary or introduction paragraph verbatim.

### SECTION 2: SKILL EXTRACTION
- "skills_list": 
   - MATCH: A single string containing ALL technical tools, languages, and frameworks found in the document.
   - FORMAT: Semicolon separated (e.g. "Python; SQL; AWS").

### OUTPUT SCHEMA (STRICT JSON):
{
  "first_name": "string",
  "last_name": "string",
  "email": "string",
  "phone": "string or null",
  "linkedin": "string or null",
  "location": "string",
  "summary": "string",
  "skills_list": "string"
}
NO MARKDOWN. NO NOTES. RAW JSON ONLY.
"""

    # ==========================================
    #  CV AGENT 2: EXPERIENCE
    # ==========================================
    PROMPT_CV_EXPERIENCE = """You are a Work History Auditor.

### OBJECTIVE:
Extract professional experience into a structured list.

### EXTRACTION RULES:
1. "company": MATCH the organization name.
2. "title": MATCH the job title.
3. "start_date": MATCH the start date verbatim.
4. "end_date": MATCH the end date verbatim.
5. "achievements_list": 
   - MATCH: Every bullet point or result described in this role.
   - FORMAT: Combine into a single string separated by semicolons (;).
   - REJECT: General duties (e.g. "Responsible for coding"). Focus on results.
6. "tools_used":
   - MATCH: Specific tools/tech mentioned *within this role description*.
   - FORMAT: Semicolon separated.

### OUTPUT SCHEMA (STRICT JSON):
{
  "items": [
    {
      "company": "string",
      "title": "string",
      "start_date": "string",
      "end_date": "string",
      "description": "string (Short summary paragraph)",
      "achievements_list": "string",
      "tools_used": "string"
    }
  ]
}
NO MARKDOWN. NO NOTES. RAW JSON ONLY.
"""

    # ==========================================
    #  CV AGENT 3: EDUCATION
    # ==========================================
    PROMPT_CV_EDUCATION = """You are an Academic Researcher.

### OBJECTIVE:
Extract education history.

### EXTRACTION RULES:
1. "institution": MATCH the university or school name.
2. "degree": MATCH the degree type (e.g., BS, MS, PhD).
3. "field": MATCH the major or field of study.
4. "details_list":
   - MATCH: Honors, GPA, Awards, or Thesis titles.
   - FORMAT: Semicolon separated string.

### OUTPUT SCHEMA (STRICT JSON):
{
  "items": [
    {
      "institution": "string",
      "degree": "string",
      "field": "string",
      "start_date": "string",
      "end_date": "string",
      "details_list": "string"
    }
  ]
}
NO MARKDOWN. RAW JSON ONLY.
"""

    # ==========================================
    #  CV AGENT 4: PROJECTS (Relational)
    # ==========================================
    PROMPT_CV_PROJECTS = """You are a Portfolio Analyst.

### OBJECTIVE:
Extract independent projects and identify their source context.

### EXTRACTION RULES:
1. "title": MATCH the project name.
2. "related_context":
   - MATCH: The name of the Company, University, or Hobby this project belongs to.
   - RULE: If it is a personal project, set to "Personal".
3. "achievements_list": Semicolon separated results.
4. "tools_used": Semicolon separated tools.

### OUTPUT SCHEMA (STRICT JSON):
{
  "items": [
    {
      "title": "string",
      "description": "string",
      "achievements_list": "string",
      "tools_used": "string",
      "related_context": "string"
    }
  ]
}
NO MARKDOWN. RAW JSON ONLY.
"""

    def fast_parse_cv(self, cv_text: str, cv_name: str = "Imported CV") -> CV:
        """
        Orchestrates the CV extraction.
        INCLUDES: Data Sanitization (Fixes Pydantic Crash) + Defensive Parsing (Fixes Empty JSON).
        """
        log.info("🚀 Starting Fast CV Parse...")
        
        # 1. Run Inference (Sequential)
        # We assume _run_inference handles the cleaning. 
        # Adding "\nOutput:" forces the model to start JSON immediately.
        identity_data = self._run_inference_cv(self.PROMPT_CV_IDENTITY + "\nOutput:", cv_text)
        exp_data = self._run_inference_cv(self.PROMPT_CV_EXPERIENCE + "\nOutput:", cv_text)
        edu_data = self._run_inference_cv(self.PROMPT_CV_EDUCATION + "\nOutput:", cv_text)
        proj_data = self._run_inference_cv(self.PROMPT_CV_PROJECTS + "\nOutput:", cv_text)

        # =========================================================
        # FIX 1: Sanitize Contact Info (Prevents Pydantic Crash)
        # =========================================================
        raw_contact = {
            "email": identity_data.get("email"),
            "phone": identity_data.get("phone"),
            "linkedin": identity_data.get("linkedin"),
            "location": identity_data.get("location")
        }
        # Filter out None/Null values so we don't break Dict[str, str]
        clean_contact = {k: str(v) for k, v in raw_contact.items() if v and v != "null"}

        # =========================================================
        # FIX 2: Create CV Object with Fallbacks
        # =========================================================
        cv_obj = CV.create(
            name=cv_name,
            first_name=identity_data.get("first_name") or "Unknown",
            last_name=identity_data.get("last_name") or "Candidate",
            title=identity_data.get("title") or "Professional",
            summary=identity_data.get("summary") or "",
            contact_info=clean_contact 
        )

        # --- Helper: Skill Dedup ---
        skill_registry = {} 

        def register_skills(raw_string):
            if not raw_string or not isinstance(raw_string, str): return []
            found_ids = []
            for s_name in raw_string.split(';'):
                clean_name = s_name.strip()
                if len(clean_name) < 2: continue
                key = clean_name.lower()
                if key not in skill_registry:
                    existing = next((s for s in cv_obj.skills if s.name.lower() == key), None)
                    if existing:
                        skill_registry[key] = existing.id
                    else:
                        new_sk = cv_obj.add_skill(name=clean_name)
                        skill_registry[key] = new_sk.id
                found_ids.append(skill_registry[key])
            return list(set(found_ids))

        # 3. Process Identity Skills
        register_skills(identity_data.get("skills_list", ""))

        # 4. Process Experience (Defensive)
        raw_exps = exp_data.get("items", [])
        if isinstance(raw_exps, list):
            for item in raw_exps:
                exp = cv_obj.add_experience(
                    title=item.get("title") or "Role",
                    company=item.get("company") or "Company",
                    start_date=item.get("start_date"),
                    end_date=item.get("end_date"),
                    description=item.get("description")
                )
                exp.skill_ids.extend(register_skills(item.get("tools_used")))
                
                # Semicolon Splitter for Achievements
                raw_ach = item.get("achievements_list", "")
                if raw_ach and isinstance(raw_ach, str):
                    for ach_text in raw_ach.split(';'):
                        if len(ach_text.strip()) > 3:
                            ach = cv_obj.add_achievement(text=ach_text.strip(), context=item.get("company"))
                            exp.add_achievement(ach)

        # 5. Process Education (Defensive)
        raw_edus = edu_data.get("items", [])
        if isinstance(raw_edus, list):
            for item in raw_edus:
                edu = cv_obj.add_education(
                    institution=item.get("institution") or "Institution",
                    degree=item.get("degree") or "Degree",
                    field=item.get("field"),
                    start_date=item.get("start_date"),
                    end_date=item.get("end_date")
                )
                # Treat details as achievements
                raw_details = item.get("details_list", "")
                if raw_details and isinstance(raw_details, str):
                    for det_text in raw_details.split(';'):
                        if len(det_text.strip()) > 3:
                            ach = cv_obj.add_achievement(text=det_text.strip(), context=item.get("institution"))
                            edu.add_achievement(ach)

        # 6. Process Projects (Relational)
        raw_projs = proj_data.get("items", [])
        if isinstance(raw_projs, list):
            for item in raw_projs:
                rel_exp_ids = []
                rel_edu_ids = []
                context = item.get("related_context", "").lower() if item.get("related_context") else ""
                
                # Simple Fuzzy Match for Relations
                if context:
                    for e in cv_obj.experiences:
                        if e.company.lower() in context: rel_exp_ids.append(e.id)
                    for ed in cv_obj.education:
                        if ed.institution.lower() in context: rel_edu_ids.append(ed.id)

                proj = cv_obj.add_project(
                    title=item.get("title") or "Project",
                    description=item.get("description") or "",
                    related_experience_ids=rel_exp_ids,
                    related_education_ids=rel_edu_ids
                )
                proj.skill_ids.extend(register_skills(item.get("tools_used")))
                
                raw_ach = item.get("achievements_list", "")
                if raw_ach and isinstance(raw_ach, str):
                    for ach_text in raw_ach.split(';'):
                        if len(ach_text.strip()) > 3:
                            ach = cv_obj.add_achievement(text=ach_text.strip(), context="Project")
                            proj.add_achievement(ach)

        log.info(f"✅ Fast Parse Complete: {len(cv_obj.experiences)} exps, {len(cv_obj.skills)} skills.")
        return cv_obj
    



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