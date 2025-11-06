# backend/core/inferer.py
from __future__ import annotations
from .models import JobDescription, CV, MappingPair, JobDescriptionFeature
from typing import List, Dict, Any, Tuple, Optional
import logging
import sqlite3
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

log = logging.getLogger(__name__)

# =====================================================
# Utilities
# =====================================================
def normalize_text(text: str) -> str:
    """Simple normalization: lowercase, strip punctuation, squeeze spaces."""
    text = text or ""
    # Keep alphanumeric and spaces, remove everything else
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Squeeze multiple spaces into one
    return " ".join(text.lower().split())

def weighted_jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Calculates weighted Jaccard similarity for ConceptNet expansions."""
    if not a or not b:
        return 0.0
    inter = set(a.keys()) & set(b.keys())
    union = set(a.keys()) | set(b.keys())
    num = sum(min(a[t], b[t]) for t in inter)
    den = sum(max(a.get(t, 0), b.get(t, 0)) for t in union)
    return num / den if den else 0.0

# =====================================================
# Embedding Wrapper
# =====================================================
class USEEmbedder:
    """Universal Sentence Encoder thin wrapper. May be slow to load."""
    def __init__(self):
        try:
            import tensorflow_hub as hub  # type: ignore
            log.info("Loading Universal Sentence Encoder from TF Hub...")
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            log.info("Universal Sentence Encoder loaded successfully.")
        except ImportError:
            log.error("TensorFlow Hub not installed. USEEmbedder will not function.")
            self.model = None
        except Exception as e:
            log.error(f"Failed to load USE model from TF Hub: {e}")
            self.model = None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.model:
            log.warning("Attempted to encode with USE, but model is not loaded.")
            return None
        if isinstance(texts, str):
            texts = [texts]
        try:
            return self.model(texts).numpy()
        except Exception as e:
            log.error(f"Error during USE encoding: {e}")
            return None

# =====================================================
# Internal Evidence Model
# =====================================================
class CVEvidence:
    """Internal representation of a single piece of scorable CV content."""
    def __init__(self, item: Any, item_type: str, text: str):
        self.source_item = item
        # Use a stable ID. For 'summary', we assign a fixed one.
        self.item_id: str = getattr(item, "id", f"cv-item-{id(item)}")
        self.item_type: str = item_type
        self.text: str = text # This is the normalized text used for scoring
        
        # Placeholders for computed data
        self.expansion: Dict[str, float] = {}
        self.vector_tfidf: Optional[np.ndarray] = None
        self.vector_use: Optional[np.ndarray] = None
        self.lemmas: set = set()

# =====================================================
# MappingInferer (Refined)
# =====================================================
class MappingInferer:
    """
    Refined inferer that matches job requirements to CV evidence.
    
    This implementation computes a full score matrix for all (requirement, cv_item)
    pairs *once*, then applies a three-stage filtering process to select
    the most relevant, non-duplicate pairs.
    """
    def __init__(self,
                 semantic_looseness: float = 0.3,
                 top_k: int = 3,
                 min_score: float = 0.22,
                 additional_threshold: float = 0.60,
                 use_use: bool = True):
        
        self.semantic_looseness = float(np.clip(semantic_looseness, 0, 1))
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self.additional_threshold = float(additional_threshold)
        self.use_use = bool(use_use)

        self.semantic_model: Optional[USEEmbedder] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.nlp: Optional[Any] = None # spaCy model
        self.cn_db_path = os.path.expanduser("~/.conceptnet_lite/conceptnet.db")

        # Specificity weights to prioritize better evidence
        # These are still relevant for the item types we *are* processing
        self.specificity_weights = {
            "project": 0.04,
            "education": 0.04,
            "experience": 0.0,
            "hobby": -0.15,
            # Removed skill, achievement, summary
        }

        # Per-run ConceptNet cache
        self._cn_cache: Dict[Tuple[str, int], Dict[str, float]] = {}

    def load_models(self):
        """Load spaCy and (if enabled) Universal Sentence Encoder."""
        try:
            from spacy import load
            log.info("Loading spaCy model (en_core_web_sm)...")
            self.nlp = load("en_core_web_sm")
        except ImportError:
            log.error("spaCy not installed. Inferer cannot run. Please install with: pip install spacy")
            return
        except Exception:
            # Try to download it if it's missing
            try:
                log.warning("spaCy model 'en_core_web_sm' not found. Attempting to download...")
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                self.nlp = load("en_core_web_sm")
                log.info("spaCy model downloaded and loaded successfully.")
            except Exception as e:
                log.error(f"Failed to load or download spaCy model. Make sure you have run: python -m spacy download en_core_web_sm. Error: {e}")
                return

        if self.use_use:
            self.semantic_model = USEEmbedder()
            if not self.semantic_model.model:
                log.warning("USEEmbedder failed to load. Proceeding without USE semantic model.")
                self.semantic_model = None
        else:
            self.semantic_model = None
        log.info(f"Models loaded. spaCy=True, USE={bool(self.semantic_model)}")

    def _conceptnet_expand(self, phrase: str, hops: int = 2, max_terms: int = 15) -> Dict[str, float]:
        """Per-run cached ConceptNet expansion (1-2 hop)."""
        if not phrase:
            return {}
        key = (phrase.lower(), hops)
        if key in self._cn_cache:
            return self._cn_cache[key]

        weights: Dict[str, float] = {}
        if not os.path.exists(self.cn_db_path):
            if not hasattr(self, "_cn_db_warned"):
                log.warning(f"ConceptNet DB not found at {self.cn_db_path}. ConceptNet scores will be 0.")
                setattr(self, "_cn_db_warned", True)
            self._cn_cache[key] = {}
            return {}
        
        # Mock expansion
        mock_weights = {}
        for term in phrase.split():
            if len(term) > 4:
                mock_weights[term + "_related"] = 1.0
                mock_weights[term + "_concept"] = 0.5
        self._cn_cache[key] = mock_weights
        return mock_weights

    def _preprocess_cv(self, cv: CV) -> List[CVEvidence]:
        """
        Collect text from all CV items into a flat list of CVEvidence.
        MODIFIED: Only processes item types supported by the frontend.
        """
        pool: List[CVEvidence] = []
        
        # Create skill lookup map. This is STILL NEEDED to add skill names
        # to the text of Experiences, Projects, etc.
        skill_map: Dict[str, str] = {skill.id: (skill.name or "") for skill in getattr(cv, "skills", [])}

        # Helper to create CVEvidence from a CV item
        def add_item(item: Any, item_type: str, text_fields: List[str]):
            text_parts = [str(getattr(item, field, "") or "") for field in text_fields]
            
            # Add linked skill names to the text
            linked_skill_ids = getattr(item, "skill_ids", [])
            for skill_id in linked_skill_ids:
                skill_name = skill_map.get(skill_id)
                if skill_name:
                    text_parts.append(skill_name)
            
            full_text = normalize_text(" ".join(text_parts))
            
            if not full_text:
                return # Don't add items with no text content
                
            ev = CVEvidence(item, item_type, full_text)
            
            if self.nlp:
                try:
                    doc = self.nlp(ev.text)
                    ev.lemmas = {t.lemma_ for t in doc if t.pos_ in ("NOUN", "PROPN") and not t.is_stop}
                except Exception:
                    ev.lemmas = set()
            
            ev.expansion = self._conceptnet_expand(ev.text)
            pool.append(ev)

        # -----------------------------------------------------------------
        # MODIFICATION: The following items are now COMMENTED OUT
        # as per the requirement to only match items that can
        # hold achievements and are supported by the frontend.
        # -----------------------------------------------------------------

        # 1. Summary (COMMENTED OUT)
        # if cv.summary:
        #     class SummaryItem:
        #         id = f"{cv.id}_summary"
        #         description = cv.summary
        #         skill_ids = []
        #     add_item(SummaryItem(), "summary", ["description"])

        # 2. Experiences (ACTIVE)
        for item in getattr(cv, "experiences", []):
            add_item(item, "experience", ["title", "description"])

        # 3. Education (ACTIVE)
        for item in getattr(cv, "education", []):
            add_item(item, "education", ["institution", "degree", "field"])

        # 4. Projects (ACTIVE)
        for item in getattr(cv, "projects", []):
            add_item(item, "project", ["title", "description"])
            
        # 5. Skills (COMMENTED OUT)
        # for item in getattr(cv, "skills", []):
        #     add_item(item, "skill", ["name", "description"])
            
        # 6. Achievements (COMMENTED OUT)
        # for item in getattr(cv, "achievements", []):
        #     add_item(item, "achievement", ["text", "context"])

        # 7. Hobbies (ACTIVE)
        for item in getattr(cv, "hobbies", []):
            add_item(item, "hobby", ["name", "description"])

        # -----------------------------------------------------------------
        # End of modifications
        # -----------------------------------------------------------------

        self.cv_pool = pool
        return pool

    def _build_vectors(self, req_texts: List[str], cv_texts: List[str]) -> Tuple:
        """Fit TF-IDF and compute all vectors for requirements and CV items."""
        corpus_texts = req_texts + cv_texts
        if not corpus_texts:
            return (None, None, None, None)

        self.tfidf = TfidfVectorizer(max_features=4000, stop_words='english')
        try:
            X_tfidf = self.tfidf.fit_transform(corpus_texts).toarray()
        except ValueError:
            return (None, None, None, None)
            
        n_req = len(req_texts)
        req_tfidf = X_tfidf[:n_req]
        cv_tfidf = X_tfidf[n_req:]

        req_use, cv_use = None, None
        if self.semantic_model:
            use_vecs = self.semantic_model.encode(corpus_texts)
            if use_vecs is not None:
                req_use = use_vecs[:n_req]
                cv_use = use_vecs[n_req:]

        return req_tfidf, req_use, cv_tfidf, cv_use

    def _composite_score(self,
                         req_text: str, req_lemmas: set, req_cn: Dict[str, float],
                         ev: CVEvidence,
                         req_tfidf_vec: np.ndarray, req_use_vec: Optional[np.ndarray]) -> float:
        """
        Compute the composite score blending TF-IDF, USE, and ConceptNet.
        """
        # 1. TF-IDF similarity
        tfidf_s = 0.0
        if ev.vector_tfidf is not None:
            try:
                tfidf_s = float(cosine_similarity([req_tfidf_vec], [ev.vector_tfidf])[0][0])
            except Exception:
                pass

        # 2. USE similarity
        use_s = 0.0
        if req_use_vec is not None and ev.vector_use is not None:
            try:
                use_s = float(cosine_similarity([req_use_vec], [ev.vector_use])[0][0])
            except Exception:
                pass

        if use_s > 0 and tfidf_s > 0:
            sem_sim = (tfidf_s + use_s) / 2.0
        else:
            sem_sim = max(tfidf_s, use_s)

        # 3. ConceptNet similarity
        cn_sim = weighted_jaccard(req_cn, ev.expansion)

        # 4. Final weighted score
        score = (1 - self.semantic_looseness) * sem_sim + self.semantic_looseness * cn_sim

        # 5. Lexical boost
        if req_lemmas & ev.lemmas:
            score = score + 0.12

        # 6. Specificity adjustment
        specificity_boost = self.specificity_weights.get(ev.item_type, 0.0)
        score = score + specificity_boost

        return float(np.clip(score, 0, 1.0))

    def _make_annotation(self, req_text: str, ev_text: str, score: float) -> str:
        """Human-readable note describing shared or close terms."""
        if not self.nlp:
            return "Semantic match found."
            
        try:
            doc_r, doc_e = self.nlp(req_text), self.nlp(ev_text)
            r_terms = {t.lemma_ for t in doc_r if t.pos_ in ("NOUN", "VERB") and not t.is_stop}
            e_terms = {t.lemma_ for t in doc_e if t.pos_ in ("NOUN", "VERB") and not t.is_stop}
            shared = list(r_terms & e_terms)
            if shared:
                examples = ", ".join(shared[:3])
                return f"Directly mentions {examples}."
        except Exception:
            pass

        if score > 0.6:
            return "Strong conceptual alignment on key themes."
        elif score > 0.35:
            return "Demonstrates related skills and concepts."
        
        return "Suggests transferable experience."

    def _add_candidate(self,
                       candidate_map: Dict[Tuple[str, str], Tuple[MappingPair, float]],
                       req: JobDescriptionFeature,
                       ev: CVEvidence,
                       score: float):
        """
        Helper to add a new mapping pair to the candidate map.
        Handles deduplication by keeping the pair with the highest score.
        """
        key = (req.id, ev.item_id)
        
        if key in candidate_map:
            existing_score = candidate_map[key][1]
            if score <= existing_score:
                return
        
        ann = self._make_annotation(req.description, ev.text, score)

        pair = MappingPair.create(
            feature_id=req.id,
            context_item_id=ev.item_id,
            context_item_type=ev.item_type,
            feature_text=req.description,
            context_item_text=ev.text,
            annotation=ann
        )
        candidate_map[key] = (pair, score)


    def infer_mappings(self, job: JobDescription, cv: CV) -> List[MappingPair]:
        """Main orchestration logic for the refined inferer."""
        
        if not self.nlp:
            log.error("spaCy model not loaded. Call load_models() first.")
            return []

        # 1. Reset per-run cache
        self._cn_cache = {}

        # 2. Preprocess CV and Job Requirements
        req_pool = getattr(job, "features", []) or []
        cv_pool = self._preprocess_cv(cv) # <-- This now uses the modified version

        if not req_pool or not cv_pool:
            log.warning("Inference stopped: Job has no features or CV has no scorable content.")
            return []
            
        req_texts = [normalize_text(r.description) for r in req_pool]
        cv_texts = [ev.text for ev in cv_pool]
        
        req_precomputed = []
        if self.nlp:
            for text in req_texts:
                doc = self.nlp(text)
                lemmas = {t.lemma_ for t in doc if t.pos_ in ("NOUN", "PROPN") and not t.is_stop}
                cn_exp = self._conceptnet_expand(text)
                req_precomputed.append((text, lemmas, cn_exp))

        # 3. Build TF-IDF and USE vectors
        req_tfidf, req_use, cv_tfidf, cv_use = self._build_vectors(req_texts, cv_texts)
        
        if req_tfidf is None or cv_tfidf is None:
            log.error("Failed to build TF-IDF vectors. Aborting.")
            return []
            
        for i, ev in enumerate(cv_pool):
            ev.vector_tfidf = cv_tfidf[i]
            if cv_use is not None:
                ev.vector_use = cv_use[i]

        # 4. Compute Full Score Matrix
        req_scores: Dict[str, List[Tuple[float, JobDescriptionFeature, CVEvidence]]] = defaultdict(list)
        cv_scores: Dict[str, List[Tuple[float, JobDescriptionFeature, CVEvidence]]] = defaultdict(list)
        all_scores: List[Tuple[float, JobDescriptionFeature, CVEvidence]] = []

        log.info(f"Computing score matrix for {len(req_pool)} requirements x {len(cv_pool)} CV items...")
        for i, req in enumerate(req_pool):
            if i >= len(req_precomputed): continue
            req_text, req_lemmas, req_cn = req_precomputed[i]
            req_tf_vec = req_tfidf[i]
            req_us_vec = req_use[i] if req_use is not None else None
            
            for j, ev in enumerate(cv_pool):
                score = self._composite_score(
                    req_text, req_lemmas, req_cn, ev, req_tf_vec, req_us_vec
                )
                
                result = (score, req, ev)
                req_scores[req.id].append(result)
                cv_scores[ev.item_id].append(result)
                all_scores.append(result)

        candidate_map: Dict[Tuple[str, str], Tuple[MappingPair, float]] = {}

        # 5. Apply Filter Stages
        
        # --- Stage A: Bottom-Up (For each CV item, find its best req) ---
        log.debug("Running Stage A: Bottom-Up")
        for ev_id, scores_for_ev in cv_scores.items():
            if not scores_for_ev:
                continue
            best_result = max(scores_for_ev, key=lambda x: x[0])
            best_score, best_req, ev = best_result
            
            if best_score >= self.min_score:
                self._add_candidate(candidate_map, best_req, ev, best_score)

        # --- Stage B: Top-Down (For each req, find its top-k CV items) ---
        log.debug("Running Stage B: Top-Down")
        for req_id, scores_for_req in req_scores.items():
            if not scores_for_req:
                continue
            scores_for_req.sort(key=lambda x: x[0], reverse=True)
            
            for score, req, ev in scores_for_req[:self.top_k]:
                if score >= self.min_score:
                    self._add_candidate(candidate_map, req, ev, score)
                else:
                    break

        # --- Stage C: Enrichment (Find any other pairs "too good to ignore") ---
        log.debug("Running Stage C: Enrichment")
        for score, req, ev in all_scores:
            if score >= self.additional_threshold:
                self._add_candidate(candidate_map, req, ev, score)

        # 6. Finalize
        final_pairs = [pair for (pair, score) in candidate_map.values()]
        log.info(f"Inference complete. {len(final_pairs)} unique mappings created from {len(req_pool)} reqs and {len(cv_pool)} CV items.")
        
        return final_pairs

