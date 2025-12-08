from __future__ import annotations
from .models import JobDescription, CV, MappingPair, JobDescriptionFeature, Skill
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
    """
    Exact architecture used in your training script.
    Required to load the state_dict correctly.
    """
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=False,
            use_safetensors=False
        )
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits

class JobSegmentEmbedder:
    """
    Wraps the fine-tuned DeBERTa model but acts as an Encoder.
    It strips the classification head logic and returns the [CLS] embedding
    from the backbone, leveraging your domain-specific training.
    """
    def __init__(self, checkpoint_path: str, model_name: str = "microsoft/deberta-v3-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        log.info(f"Loading custom Job Segment Model from {checkpoint_path}...")
        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 2. Load Model Architecture
            # Hardcoded 12 classes as defined in your training script TAXONOMY
            self.model = DebertaMultiLabel(model_name, num_classes=12)
            
            # 3. Load State Dict
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                # Handle case where checkpoint saves 'model_state' key or full dict
                state_dict = checkpoint.get('model_state', checkpoint)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval() # Set to eval mode (disable dropout)
                log.info(f"Custom DeBERTa Model loaded successfully on {self.device}.")
            else:
                log.error(f"Checkpoint not found at {checkpoint_path}. Semantic matching will be disabled.")
                self.model = None

        except Exception as e:
            log.error(f"Failed to load custom model: {e}", exc_info=True)
            self.model = None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Returns numpy array of embeddings (shape: [N, hidden_size]).
        """
        if not self.model or not texts:
            return None
            
        embeddings = []
        batch_size = 16 # Keep small for CPU safety
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                encoded_input = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=128, 
                    return_tensors='pt'
                )
                
                input_ids = encoded_input['input_ids'].to(self.device)
                attention_mask = encoded_input['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    # Pass through backbone only
                    outputs = self.model.backbone(input_ids=input_ids, attention_mask=attention_mask)
                    # Extract [CLS] token (first token of last hidden state)
                    # This vector contains the "semantic summary" the model learned
                    cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                embeddings.append(cls_vectors)
            except Exception as e:
                log.error(f"Encoding error on batch {i}: {e}")
                return None

        if embeddings:
            return np.vstack(embeddings)
        return None

# =====================================================
# 2. Atomic Evidence Model
# =====================================================
class CVEvidence:
    """
    Represents a Micro-Segment of a CV Entity.
    """
    def __init__(self, 
                 parent_item: Any, 
                 item_type: str, 
                 segment_text: str, 
                 parent_context: str,
                 parent_title: str):
        
        self.parent_item = parent_item
        # This ID ties all segments together to the parent entity
        self.parent_id: str = getattr(parent_item, "id", f"cv-{item_type}-{hash(parent_title)}")
        
        self.item_type: str = item_type 
        self.text: str = segment_text
        self.parent_context: str = parent_context 
        self.parent_title: str = parent_title

        self.vector_tfidf: Optional[np.ndarray] = None
        self.vector_embedding: Optional[np.ndarray] = None

# =====================================================
# 3. MappingInferer (Aggregated Logic)
# =====================================================
class MappingInferer:
    def __init__(self):
        # Default tuning parameters (will be overridden by API kwargs)
        self.default_min_score = 0.28
        self.default_top_k = 3
        
        # Model Components
        self.embedder: Optional[JobSegmentEmbedder] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.nlp: Optional[Any] = None
        
        # Path to your trained model checkpoint
        self.CHECKPOINT_PATH = "backend/core/job_model/checkpoints/model_best.pt"

        # Type Matrix: Boost score if the Requirement Type matches the Evidence Type
        self.type_matrix = {
            ("hard_skill", "skill"): 0.20,
            ("soft_skill", "skill"): 0.15,
            ("responsibility", "achievement"): 0.15, 
            ("hard_skill", "achievement"): 0.15, # Skills proven by achievements
            ("responsibility", "experience"): 0.10,
            ("hard_skill", "project"): 0.10,
            ("qualification", "education"): 0.15,
            ("qualification", "certification"): 0.15,
            ("benefit", "hobby"): 0.05
        }

    def load_models(self):
        """Called by main.py startup."""
        log.info("Loading spaCy model...")
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                log.warning(f"Could not load spaCy: {e}. Lemmatization disabled.")

        log.info("Initializing JobSegmentEmbedder...")
        self.embedder = JobSegmentEmbedder(self.CHECKPOINT_PATH)

    def _preprocess_cv(self, cv: CV) -> List[CVEvidence]:
        """
        Deconstructs CV into ISOLATED ATOMIC pieces.
        Prioritizes Titles and Names as standalone vectors.
        """
        pool: List[CVEvidence] = []
        
        # Lookup maps
        skill_map = {s.id: s for s in getattr(cv, "skills", [])}
        ach_map = {a.id: a for a in getattr(cv, "achievements", [])}

        # --- Helper: Safely add children (Achievements/Skills) ---
        def add_children(parent_obj, parent_type, parent_title):
            # 1. Achievements
            ach_ids = getattr(parent_obj, "achievement_ids", [])
            for ach_id in ach_ids:
                ach = ach_map.get(ach_id)
                if ach and getattr(ach, 'text', None):
                    # Combine text + context
                    context_str = getattr(ach, 'context', '') or ''
                    display_text = f"{ach.text} {context_str}".strip()
                    # We do NOT inject parent title here anymore to keep signal pure for keywords,
                    # but we keep it in metadata for the note.
                    pool.append(CVEvidence(parent_obj, "achievement", normalize_text(display_text), parent_type, parent_title))

            # 2. Linked Skills
            sk_ids = getattr(parent_obj, "skill_ids", [])
            for sk_id in sk_ids:
                sk = skill_map.get(sk_id)
                if sk and getattr(sk, 'name', None):
                    # Pure Name
                    pool.append(CVEvidence(parent_obj, "skill", normalize_text(sk.name), parent_type, parent_title))

        # --- 1. Experiences ---
        for exp in getattr(cv, "experiences", []):
            title = getattr(exp, 'title', '')
            company = getattr(exp, 'company', '')
            display_title = f"{title} at {company}"

            # A. ISOLATED TITLE (The "Role" Signal)
            if title:
                pool.append(CVEvidence(exp, "experience", normalize_text(title), "Work History", display_title))
            
            # B. Description Segments
            desc = getattr(exp, 'description', '') or ''
            segments = intelligent_segmentation(desc, self.nlp)
            for seg in segments:
                # Keep segment pure, don't mix with title in vector text
                pool.append(CVEvidence(exp, "experience", normalize_text(seg), "Work History", display_title))
            
            add_children(exp, "Work History", display_title)

        # --- 2. Projects ---
        for proj in getattr(cv, "projects", []):
            title = getattr(proj, 'title', '')
            
            # A. ISOLATED TITLE
            if title:
                pool.append(CVEvidence(proj, "project", normalize_text(title), "Project", title))

            # B. Description Segments
            desc = getattr(proj, 'description', '') or ''
            segments = intelligent_segmentation(desc, self.nlp)
            for seg in segments:
                pool.append(CVEvidence(proj, "project", normalize_text(seg), "Project", title))
            
            add_children(proj, "Project", title)

        # --- 3. Education ---
        for edu in getattr(cv, "education", []):
            deg = getattr(edu, 'degree', '')
            field = getattr(edu, 'field', '')
            inst = getattr(edu, 'institution', '')
            
            display_title = f"{deg} - {inst}"
            
            # A. ISOLATED DEGREE + FIELD
            core_text = f"{deg} {field}".strip()
            if core_text:
                pool.append(CVEvidence(edu, "education", normalize_text(core_text), "Education", display_title))
            
            # B. Full Text (Degree + Field + Institution)
            full_text = f"{deg} in {field} from {inst}"
            pool.append(CVEvidence(edu, "education", normalize_text(full_text), "Education", display_title))

        # --- 4. Hobbies ---
        for hobby in getattr(cv, "hobbies", []):
            name = getattr(hobby, 'name', '')
            desc = getattr(hobby, 'description', '') or ''
            
            if name:
                pool.append(CVEvidence(hobby, "hobby", normalize_text(name), "Hobby", name))
            if desc:
                 pool.append(CVEvidence(hobby, "hobby", normalize_text(desc), "Hobby", name))

        # --- 5. Global Skills (ISOLATED) ---
        for sk in getattr(cv, "skills", []):
            name = getattr(sk, 'name', '')
            if name:
                pool.append(CVEvidence(sk, "skill", normalize_text(name), "Skillset", name))

        return pool

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
        active_top_k = top_k if top_k is not None else self.default_top_k

        if not self.embedder: self.load_models()

        job_features = [f for f in getattr(job, "features", []) if f.description]
        evidence_pool = self._preprocess_cv(cv)

        if not job_features or not evidence_pool:
            return []

        req_texts = [normalize_text(f.description) for f in job_features]
        cv_texts = [normalize_text(e.text) for e in evidence_pool]

        r_tf, r_emb, c_tf, c_emb = self._build_vectors(req_texts, cv_texts)
        if r_tf is None: return []

        mappings = []
        
        for i, req in enumerate(job_features):
            
            # Aggregate scores by Parent ID
            entity_scores = defaultdict(list) 
            
            for j, ev in enumerate(evidence_pool):
                # A. Similarity
                tfidf_sim = float(cosine_similarity([r_tf[i]], [c_tf[j]])[0][0])
                emb_sim = 0.0
                if r_emb is not None and c_emb is not None:
                    emb_sim = float(cosine_similarity([r_emb[i]], [c_emb[j]])[0][0])
                
                # --- SCORING STRATEGY (V4 - Strict) ---
                if tfidf_sim > 0.7:
                    base_score = tfidf_sim
                else:
                    # Weighted average: 30% Keyword, 70% Semantic
                    base_score = (0.3 * tfidf_sim) + (0.7 * emb_sim) if emb_sim else tfidf_sim

                # --- SMART SAFEGUARD ---
                # 1. Hard Skills / Qualifications: ZERO Tolerance for missing keywords
                # This fixes "Striker" matching "AWS" (Skill).
                if req.type in ["hard_skill", "qualification", "tool"]:
                    if tfidf_sim < 0.1: 
                        base_score = 0.0 # KILL IT.
                
                # 2. General Safeguard: Penalty for pure hallucinations
                # This fixes "Maintain App" matching "Architect Library" (Responsibility).
                elif tfidf_sim < 0.05: 
                    base_score = base_score * 0.4 # Heavy penalty

                # B. Type Bonus (Only applied if the base score survives)
                type_bonus = 0.0
                if base_score > 0.35:
                     type_bonus = self.type_matrix.get((req.type, ev.item_type), 0.0)
                
                final_score = min(base_score + type_bonus, 1.0)
                
                if final_score >= active_min_score:
                    entity_scores[ev.parent_id].append({
                        "score": final_score,
                        "segment_text": ev.text,
                        "evidence_obj": ev
                    })

            # Aggregation & Note Generation
            final_candidates = []
            
            for parent_id, hits in entity_scores.items():
                best_segment_score = max(h["score"] for h in hits)
                hits.sort(key=lambda x: x["score"], reverse=True)
                top_hits = hits[:2] 
                primary_ev = top_hits[0]["evidence_obj"]
                
                # --- NOTE CLEANUP ---
                if primary_ev.item_type.lower() == primary_ev.parent_context.lower().replace(" ", ""):
                     note = f"Matched via {primary_ev.item_type.title()}: \"{primary_ev.parent_title}\". "
                elif primary_ev.item_type == "experience" and primary_ev.parent_context == "Work History":
                     note = f"Matched via Experience: \"{primary_ev.parent_title}\". "
                else:
                     note = f"Matched via {primary_ev.item_type.title()} in {primary_ev.parent_context} ({primary_ev.parent_title}). "
                
                details = []
                for h in top_hits:
                    txt = h['segment_text']
                    if len(txt) > 100: txt = txt[:97] + "..."
                    # Avoid repeating title in excerpts
                    if txt.lower() not in primary_ev.parent_title.lower(): 
                        details.append(f"'{txt}'")
                
                if details:
                    note += f"Relevant excerpts: {'; '.join(details)}."
                
                confidence_pct = int(best_segment_score * 100)
                note += f" (Match Confidence: {confidence_pct}%)"
                
                if best_segment_score > 0.85:
                    note += " - Strong Match"

                final_candidates.append({
                    "score": best_segment_score,
                    "evidence": primary_ev,
                    "note": note
                })

            final_candidates.sort(key=lambda x: x["score"], reverse=True)
            
            for candidate in final_candidates[:active_top_k]:
                ev = candidate["evidence"]
                pair = MappingPair.create(
                    feature_id=req.id,
                    context_item_id=ev.parent_id,
                    context_item_type=ev.item_type,
                    feature_text=req.description,
                    context_item_text=ev.parent_title, 
                    strength=candidate["score"],
                    annotation=candidate["note"] 
                )
                mappings.append(pair)

        return mappings