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


from typing import TYPE_CHECKING

# We only import for type hinting, avoiding circular dependencies at runtime
if TYPE_CHECKING:
    from .llm_manager import LLMManager


log = logging.getLogger(__name__)

class JobParser:
    """
    Handles logic for parsing Job Descriptions.
    """
    
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
      "description": "string (The extracted text verbatim; separate with SEMICOLON if multiple found)"
    }
  ]
}
DO NOT RETURN MARKDOWN. DO NOT USE ```json BLOCKS. JUST THE RAW JSON.
"""

    def __init__(self, manager: LLMManager):
        self.manager = manager

    async def fast_parse(self, job_text: str) -> dict:
        log.info("⚡ Starting Fast Parse (Master Agent v1)...")
        start_time = time.time()

        # 1. Request Inference via Manager
        # Note: Using temperature=0.0 as per your 'job' requirement
        data = await self.manager.request_inference(
            system_prompt=self.PROMPT_MASTER,
            user_text=job_text,
            doc_label="JOB DESCRIPTION",
            is_private=True,
            temperature=0.0,
            max_tokens=8192
        )

        # 2. Logic: Initialize Result
        final_result = {
            "title": data.get("title"),
            "company": data.get("company"),
            "location": data.get("location"),
            "salary_range": data.get("salary_range"),
            "features": [],
            "_meta": {}
        }

        # 3. Logic: Deduplicate and Map Features
        raw_features = data.get("features", [])
        seen = set()
        unique_features = []

        VALID_TYPES = [
            "responsibility", "hard_skill", "soft_skill", "experience",
            "qualification", "requirement", "nice_to_have", 
            "employer_mission", "employer_culture", "role_value", 
            "benefit", "other"
        ]

        for f in raw_features:
            if not isinstance(f, dict): continue
            
            raw_val = f.get("description")
            if raw_val is None: continue 
            clean_desc_full = str(raw_val).strip()
            if not clean_desc_full: continue
            
            feat_type = f.get("type")

            # Mapping
            if feat_type == "certification": feat_type = "qualification"
            if feat_type == "perk": feat_type = "benefit"

            # Validation
            if feat_type not in VALID_TYPES and feat_type != "requirement":
                continue

            # Unbundling Logic (Semicolons)
            for raw_item in clean_desc_full.split(';'):
                clean_desc_semi = raw_item.strip()
                for raw_line in clean_desc_semi.split(','):
                    clean_desc = raw_line.strip()
                    # Final Filters
                    if not clean_desc or len(clean_desc) < 2: continue
                    if clean_desc.lower().startswith("we are looking"): continue
                    if clean_desc == final_result.get("location"): continue
                    
                    sig = (feat_type, clean_desc.lower())
                    if sig not in seen:
                        seen.add(sig)
                        unique_features.append({
                            "type": feat_type,
                            "description": clean_desc,
                            "preferred": bool(f.get("preferred", False))
                        })

        final_result["features"] = unique_features
        final_result["_meta"]["generation_time_sec"] = round(time.time() - start_time, 2)
        
        log.info(f"✅ Fast Parse Complete in {final_result['_meta']['generation_time_sec']}s")
        return final_result


class CVParser:
    """
    Handles logic for parsing CVs.
    """
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

    def __init__(self, manager: LLMManager):
        self.manager = manager

    async def parse_cv(self, cv_text: str, user_id: str, cv_name: str = "Imported CV") -> CV:
        log.info("🚀 Starting Fast CV Parse...")
        
        # 1. Run Inference (Sequential)
        identity_data = await self.manager.request_inference(
            self.PROMPT_CV_IDENTITY + "\nOutput:", cv_text, 
            doc_label="CANDIDATE RESUME", is_private=True, temperature=0.1, max_tokens=1000
        )
        exp_data = await self.manager.request_inference(
            self.PROMPT_CV_EXPERIENCE + "\nOutput:", cv_text, 
            doc_label="CANDIDATE RESUME", is_private=True, temperature=0.1, max_tokens=2500
        )
        edu_data = await self.manager.request_inference(
            self.PROMPT_CV_EDUCATION + "\nOutput:", cv_text, 
            doc_label="CANDIDATE RESUME", is_private=True, temperature=0.1, max_tokens=1000
        )
        proj_data = await self.manager.request_inference(
            self.PROMPT_CV_PROJECTS + "\nOutput:", cv_text, 
            doc_label="CANDIDATE RESUME", is_private=True, temperature=0.1, max_tokens=1000
        )

        # 2. Sanitization
        raw_contact = {
            "email": identity_data.get("email"),
            "phone": identity_data.get("phone"),
            "linkedin": identity_data.get("linkedin"),
            "location": identity_data.get("location")
        }
        clean_contact = {k: str(v) for k, v in raw_contact.items() if v and v != "null"}

        # 3. Create CV Object
        cv_obj = CV.create(
            user_id=user_id,
            name=cv_name,
            first_name=identity_data.get("first_name") or "Unknown",
            last_name=identity_data.get("last_name") or "Candidate",
            title=identity_data.get("title") or "Professional",
            summary=identity_data.get("summary") or "",
            contact_info=clean_contact 
        )

        # 4. Helper: Skill Dedup
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

        # 5. Assembly
        register_skills(identity_data.get("skills_list", ""))

        # --- Experience ---
        for item in exp_data.get("items", []) or []:
            exp = cv_obj.add_experience(
                title=item.get("title") or "Role",
                company=item.get("company") or "Company",
                start_date=item.get("start_date"),
                end_date=item.get("end_date"),
                description=item.get("description")
            )
            exp.skill_ids.extend(register_skills(item.get("tools_used")))
            
            # FIX: Explicitly add achievement instead of chaining .link_to_experience()
            if item.get("achievements_list"):
                for ach_text in str(item.get("achievements_list", "")).split(';'):
                    clean_text = ach_text.strip()
                    if len(clean_text) > 3:
                        ach = cv_obj.add_achievement(text=clean_text, context=item.get("company"))
                        exp.add_achievement(ach) # This method exists on Experience model

        # --- Education ---
        for item in edu_data.get("items", []) or []:
            edu = cv_obj.add_education(
                institution=item.get("institution") or "Institution",
                degree=item.get("degree") or "Degree",
                field=item.get("field"),
                start_date=item.get("start_date"),
                end_date=item.get("end_date")
            )
            
            # FIX: Treat details as achievements and link explicitly
            if item.get("details_list"):
                for det_text in str(item.get("details_list", "")).split(';'):
                     clean_text = det_text.strip()
                     if len(clean_text) > 3:
                        ach = cv_obj.add_achievement(text=clean_text, context=item.get("institution"))
                        edu.add_achievement(ach) # This method exists on Education model

        # --- Projects ---
        for item in proj_data.get("items", []) or []:
            rel_exp_ids = []
            rel_edu_ids = []
            ctx = (item.get("related_context") or "").lower()
            if ctx:
                for e in cv_obj.experiences:
                    if e.company.lower() in ctx: rel_exp_ids.append(e.id)
                for ed in cv_obj.education:
                    if ed.institution.lower() in ctx: rel_edu_ids.append(ed.id)

            proj = cv_obj.add_project(
                title=item.get("title") or "Project",
                description=item.get("description") or "",
                related_experience_ids=rel_exp_ids,
                related_education_ids=rel_edu_ids
            )
            proj.skill_ids.extend(register_skills(item.get("tools_used")))
            
            # FIX: Link Achievements explicitly
            if item.get("achievements_list"):
                for ach_text in str(item.get("achievements_list", "")).split(';'):
                    clean_text = ach_text.strip()
                    if len(clean_text) > 3:
                        ach = cv_obj.add_achievement(text=clean_text, context="Project")
                        proj.add_achievement(ach) # This method exists on Project model

        log.info(f"✅ Fast Parse Complete: {len(cv_obj.experiences)} exps, {len(cv_obj.skills)} skills.")
        return cv_obj


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

class MappingInfererOld:
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


from backend.core.utils.inference import MatchScoring, SmartNoteBuilder

class MappingInferer:
    def __init__(self):
        self.default_min_score = 0.28
        self.embedder: Optional[JobSegmentEmbedder] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.nlp: Optional[Any] = None
        self.CHECKPOINT_PATH = "backend/core/job_model/checkpoints/model_best.pt"
        
        # REMOVED: self.type_matrix (Moved to MatchScoring)

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
            
            # --- 1. Collect All Valid Matches (Quality Gate) ---
            candidates = []
            for j, ev in enumerate(evidence_pool):
                tfidf_sim = float(cosine_similarity([r_tf[i]], [c_tf[j]])[0][0])
                emb_sim = 0.0
                if r_emb is not None and c_emb is not None:
                    emb_sim = float(cosine_similarity([r_emb[i]], [c_emb[j]])[0][0])
                
                # REFACTOR 1: Delegate scoring to central logic
                final_score = MatchScoring.calculate_strength(
                    req_type=req.type,
                    evidence_type=ev.item_type,
                    tfidf_sim=tfidf_sim,
                    emb_sim=emb_sim
                )
                
                if final_score >= active_min_score:
                    candidates.append({"score": final_score, "evidence": ev})

            # --- 2. Group by Root Parent (Deduplication) ---
            parent_groups = defaultdict(list)
            for cand in candidates:
                root_id = cand['evidence'].occurrences[0][0]['id']
                parent_groups[root_id].append(cand)

            # --- 3. Create MappingPairs for Groups ---
            group_results = []
            
            for pid, group_hits in parent_groups.items():
                group_hits.sort(key=lambda x: x['score'], reverse=True)
                
                best_hit = group_hits[0]
                best_ev = best_hit['evidence']
                best_path = best_ev.occurrences[0] # List[Dict]

                # --- REFACTOR 2: SMART NOTE GENERATION ---
                # Delegate text formatting to the Presentation Logic
                note = SmartNoteBuilder.build(
                    text=best_ev.text,
                    score=best_hit['score'],
                    paths=best_ev.occurrences # Pass all occurrences for "Also found in..." logic
                )
                # -----------------------------------------

                # A. Build Best Match Candidate
                best_candidate = MatchCandidate(
                    segment_text=best_ev.text,
                    segment_type=best_ev.item_type,
                    score=best_hit['score'],
                    lineage=[LineageItem(**step) for step in best_path]
                )

                # B. Build Supporting Candidates
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
                    summary_note=note
                )

                # D. Calculate Pair Strength (Aggregated)
                # REFACTOR 3: Use central aggregation logic
                all_candidates_objs = [best_candidate] + supporting
                final_pair_strength = MatchScoring.aggregate_pair_strength(all_candidates_objs)

                # E. Create Mapping Pair
                root_container = best_path[0]
                
                pair = MappingPair(
                    feature_id=req.id,
                    feature_text=req.description,
                    context_item_id=root_container['id'],
                    context_item_type=root_container['type'],
                    context_item_text=root_container['name'],
                    strength=final_pair_strength, # Updated to use aggregated score
                    meta=meta_obj,
                    annotation=meta_obj.summary_note
                )
                
                group_results.append(pair)

            # 4. Final Processing: Sort and Return ALL
            group_results.sort(key=lambda x: x.strength, reverse=True)
            mappings.extend(group_results)

        return mappings