# backend/core/inferer.py
from .models import JobDescription, JobDescriptionFeature, CV, MappingPair
from typing import List, Optional, Dict, Any
import logging
# We NO LONGER import sentence_transformers or spacy here.

log = logging.getLogger(__name__)

# A simple internal data class to hold "flattened" CV data
class CVEvidence:
    """A flattened, searchable representation of a single CV item."""
    def __init__(self, item: Any, item_type: str, text: str, skills: List[str]):
        self.item_id: str = item.id
        self.item_type: str = item_type # e.g., 'experiences', 'education'
        self.source_item: Any = item # The original Pydantic model
        self.full_text: str = text # The "document" we will search
        self.skills: List[str] = [s.lower() for s in skills] # Keywords
        self.vector: Optional[Any] = None # Will hold the semantic vector

    def __repr__(self):
        return f"CVEvidence(type={self.item_type}, id={self.item_id}, text='{self.full_text[:50]}...')"

class MappingInferer:
    """
    Uses a multi-stage NLP pipeline to infer mappings between a JobDescription and a CV.
    """
    def __init__(self):
        """
        Initialize the inferer. Models are NOT loaded here to prevent
        multiprocessing deadlocks with uvicorn reloader.
        Call load_models() at app startup.
        """
        # These are now typed as Optional[Any] since the classes are not imported yet.
        self.semantic_model: Optional[Any] = None
        self.nlp: Optional[Any] = None
        self.util: Optional[Any] = None # To store the 'util' from sentence_transformers
        log.info("MappingInferer initialized (models NOT loaded).")
            
        self.cv_evidence_pool: List[CVEvidence] = []
        self.skill_map: Dict[str, str] = {} # {skill_name.lower(): skill_id}

    def load_models(self):
        """
        Loads the heavy NLP models. This should be called *once*
        at application startup (e.g., in a FastAPI startup event).
        """
        log.info("Loading NLP models for MappingInferer...")
        try:
            # --- IMPORTS ARE MOVED HERE ---
            # This is the key fix. These libraries are now imported *only*
            # by the worker process, not the reloader.
            from sentence_transformers import SentenceTransformer, util
            import spacy
            # ------------------------------

            # 1. Load a powerful model for semantic sentence comparison
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. Load a spaCy model for keyword/entity extraction
            self.nlp = spacy.load("en_core_web_sm")
            
            # 3. Store the 'util' module for later use
            self.util = util
            
            log.info("NLP models loaded successfully.")
        except ImportError:
            log.critical("Failed to import NLP libraries. Please run `pip install sentence-transformers spacy`")
            raise
        except Exception as e:
            log.error(f"Failed to load NLP models: {e}")
            log.error("Have you run `python -m spacy download en_core_web_sm`?")
            raise e

    def _check_models_loaded(self):
        """Internal check to ensure models are ready before use."""
        if not self.semantic_model or not self.nlp or not self.util:
            log.error("Inferer models are not loaded. Call load_models() at application startup.")
            raise RuntimeError("Inferer models are not loaded. Ensure load_models() is called in a FastAPI startup event.")

    def _preprocess_cv(self, cv: CV) -> List[CVEvidence]:
        """
        "Flattens" the CV into a single, searchable list of CVEvidence objects.
        """
        self._check_models_loaded() # This check is now critical
        log.info(f"Preprocessing CV {cv.id}...")
        pool: List[CVEvidence] = []
        self.skill_map = {skill.name.lower(): skill.id for skill in cv.skills}
        ach_map = {ach.id: ach for ach in cv.achievements}

        def get_item_skills(item: Any) -> List[str]:
            """Helper to get all skill names from an item and its achievements."""
            skill_ids = set(getattr(item, 'skill_ids', []))
            for ach_id in getattr(item, 'achievement_ids', []):
                if ach_id in ach_map:
                    skill_ids.update(ach_map[ach_id].skill_ids)
            
            return [s.name for s in cv.skills if s.id in skill_ids]

        # 1. Process Experiences
        for item in cv.experiences:
            ach_texts = " ".join([ach_map[ach_id].text for ach_id in item.achievement_ids if ach_id in ach_map])
            full_text = f"{item.title} at {item.company}. {item.description or ''} {ach_texts}"
            pool.append(CVEvidence(item, 'experiences', full_text, get_item_skills(item)))

        # 2. Process Education
        for item in cv.education:
            full_text = f"{item.degree} in {item.field} from {item.institution}."
            pool.append(CVEvidence(item, 'education', full_text, get_item_skills(item)))
            
        # 3. Process Projects
        for item in cv.projects:
            ach_texts = " ".join([ach_map[ach_id].text for ach_id in item.achievement_ids if ach_id in ach_map])
            full_text = f"Project: {item.title}. {item.description or ''} {ach_texts}"
            pool.append(CVEvidence(item, 'projects', full_text, get_item_skills(item)))

        # 4. Process Hobbies
        for item in cv.hobbies:
            full_text = f"Hobby: {item.name}. {item.description or ''}"
            pool.append(CVEvidence(item, 'hobbies', full_text, get_item_skills(item)))

        # 5. Pre-compute all semantic vectors for the CV
        if pool:
            all_texts = [evidence.full_text for evidence in pool]
            vectors = self.semantic_model.encode(all_texts, convert_to_tensor=True)
            for evidence, vector in zip(pool, vectors):
                evidence.vector = vector
        
        log.info(f"CV preprocessing complete. {len(pool)} evidence items vectorized.")
        return pool

    def _generate_semantic_reason(self, req_text: str, cv_text: str) -> str:
        """
        Generates a human-readable reason for a semantic match
        by extracting and comparing key topics.
        """
        self._check_models_loaded() 
        
        def get_topics(text: str, max_topics=3) -> List[str]:
            """Uses spaCy to extract key nouns and verbs."""
            doc = self.nlp(text)
            topics = [
                token.lemma_  # Use the base form (e.g., "managing" -> "manage")
                for token in doc 
                if token.pos_ in ("NOUN", "VERB") # Only get nouns & verbs
                and not token.is_stop         # Ignore words like 'a', 'the'
                and not token.is_punct        # Ignore '.' ',' etc.
            ]
            return list(dict.fromkeys(topics))[:max_topics]

        req_topics = get_topics(req_text)
        cv_topics = get_topics(cv_text)
        
        if not req_topics or not cv_topics:
            return "Semantic text similarity." # Fallback

        req_topic_str = f"'{', '.join(req_topics)}'"
        cv_topic_str = f"'{', '.join(cv_topics)}'"
        
        return f"Requirement topics (e.g., {req_topic_str}) relate to your experience with {cv_topic_str}."

    def _find_best_match(self, req: JobDescriptionFeature) -> Optional[tuple[CVEvidence, float, str]]:
        """
        This is the core pipeline. It runs strategies from most to least precise.
        """
        self._check_models_loaded() 
        
        # --- Strategy 1: Rule-Based Filtering ---
        if req.type == 'qualification':
            target_pool = [e for e in self.cv_evidence_pool if e.item_type == 'education']
        else:
            target_pool = self.cv_evidence_pool

        # --- Strategy 2: Direct Keyword/Entity Match ---
        req_doc = self.nlp(req.description)
        req_skills = [token.text.lower() for token in req_doc if token.pos_ in ("NOUN", "PROPN")]
        
        for evidence in target_pool:
            matched_skills = [s for s in req_skills if s in evidence.skills]
            if matched_skills:
                reason = f"Direct skill match for: {', '.join(matched_skills)}"
                return (evidence, 0.95, reason) 

        # --- Strategy 3: Semantic Similarity (The NLP) ---
        if not target_pool: 
            return None

        req_vector = self.semantic_model.encode(req.description, convert_to_tensor=True)
        evidence_vectors = [evidence.vector for evidence in target_pool]
        
        # Use the stored 'util' module
        cosine_scores = self.util.cos_sim(req_vector, evidence_vectors)[0]
        
        top_score_idx = cosine_scores.argmax()
        top_score = cosine_scores[top_score_idx].item()
        best_evidence = target_pool[top_score_idx]
        
        CONFIDENCE_THRESHOLD = 0.60
        
        if top_score >= CONFIDENCE_THRESHOLD:
            reason = self._generate_semantic_reason(
                req.description, 
                best_evidence.full_text
            )
            return (best_evidence, top_score, reason)

        return None

    def infer_mappings(self, job: JobDescription, cv: CV) -> List[MappingPair]:
        """
        The main public method.
        Generates a list of high-confidence MappingPair suggestions.
        """
        self.cv_evidence_pool = self._preprocess_cv(cv)
        suggestions: List[MappingPair] = []
        
        for req in job.features:
            best_match = self._find_best_match(req) 
            
            if best_match:
                evidence, score, reason = best_match
                
                item_text = "CV Item"
                if evidence.item_type == 'experiences':
                    item_text = f"{evidence.source_item.title} @ {evidence.source_item.company}"
                elif evidence.item_type == 'education':
                    item_text = f"{evidence.source_item.degree} @ {evidence.source_item.institution}"
                elif evidence.item_type == 'projects':
                    item_text = f"{evidence.source_item.title} (Project)"
                elif evidence.item_type == 'hobbies':
                    item_text = f"{evidence.source_item.name} (Hobby)"

                pair = MappingPair.create(
                    feature_id=req.id,
                    context_item_id=evidence.item_id,
                    context_item_type=evidence.item_type,
                    feature_text=req.description,
                    context_item_text=item_text,
                    annotation=f"Inferred Match (Confidence: {score*100:.0f}%): {reason}"
                )
                suggestions.append(pair)
                
        return suggestions