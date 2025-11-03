# backend/core/inferer.py
from .models import JobDescription, JobDescriptionFeature, CV, MappingPair
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer, util
import spacy
import re
import logging

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
        Initialize the inferer by loading the necessary NLP models.
        This is done once on server startup to be fast.
        """
        log.info("Loading NLP models for MappingInferer...")
        try:
            # 1. Load a powerful model for semantic sentence comparison
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. Load a spaCy model for keyword/entity extraction
            self.nlp = spacy.load("en_core_web_sm")
            log.info("NLP models loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load NLP models: {e}")
            log.error("Please run `pip install sentence-transformers spacy` and `python -m spacy download en_core_web_sm`")
            raise e
            
        self.cv_evidence_pool: List[CVEvidence] = []
        self.skill_map: Dict[str, str] = {} # {skill_name.lower(): skill_id}

    def _preprocess_cv(self, cv: CV) -> List[CVEvidence]:
        """
        "Flattens" the CV into a single, searchable list of CVEvidence objects.
        This is the most critical preprocessing step.
        """
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
            # Return a unique list of the first N topics
            return list(dict.fromkeys(topics))[:max_topics]

        req_topics = get_topics(req_text)
        cv_topics = get_topics(cv_text)
        
        # If we can't find good topics, fall back to a generic reason
        if not req_topics or not cv_topics:
            return "Semantic text similarity." # Fallback

        # Format the topics into a clear, readable string
        req_topic_str = f"'{', '.join(req_topics)}'"
        cv_topic_str = f"'{', '.join(cv_topics)}'"
        
        return f"Requirement topics (e.g., {req_topic_str}) relate to your experience with {cv_topic_str}."

    def _find_best_match(self, req: JobDescriptionFeature) -> Optional[tuple[CVEvidence, float, str]]:
        """
        This is the core pipeline. It runs strategies from most to least precise.
        """
        
        # --- Strategy 1: Rule-Based Filtering (Your "Smart" Routing) ---
        if req.type == 'qualification':
            target_pool = [e for e in self.cv_evidence_pool if e.item_type == 'education']
        else:
            target_pool = self.cv_evidence_pool

        # --- Strategy 2: Direct Keyword/Entity Match ---
        # Extract keywords from the requirement
        req_doc = self.nlp(req.description)
        req_skills = [token.text.lower() for token in req_doc if token.pos_ in ("NOUN", "PROPN")]
        
        for evidence in target_pool:
            # Check for direct skill name matches
            matched_skills = [s for s in req_skills if s in evidence.skills]
            if matched_skills:
                reason = f"Direct skill match for: {', '.join(matched_skills)}"
                return (evidence, 0.95, reason) # High confidence

        # --- Strategy 3: Semantic Similarity (The NLP) ---
        if not target_pool: # If filtering removed all candidates
            return None

        # Encode the job requirement
        req_vector = self.semantic_model.encode(req.description, convert_to_tensor=True)
        
        # Get the pre-computed vectors from our pool
        evidence_vectors = [evidence.vector for evidence in target_pool]
        
        # Compute cosine similarity
        cosine_scores = util.cos_sim(req_vector, evidence_vectors)[0]
        
        # Find the best score
        top_score_idx = cosine_scores.argmax()
        top_score = cosine_scores[top_score_idx].item()
        best_evidence = target_pool[top_score_idx]
        
        # Disqualification: Set a confidence threshold
        CONFIDENCE_THRESHOLD = 0.60
        
        if top_score >= CONFIDENCE_THRESHOLD:
            # Call our new helper to generate an explainable reason
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
            # For each requirement, find the best piece of evidence
            best_match = self_find_best_match(req)
            
            if best_match:
                evidence, score, reason = best_match
                
                # Get a simple display text for the context item
                item_text = "CV Item"
                if evidence.item_type == 'experiences':
                    item_text = f"{evidence.source_item.title} @ {evidence.source_item.company}"
                elif evidence.item_type == 'education':
                    item_text = f"{evidence.source_item.degree} @ {evidence.source_item.institution}"
                elif evidence.item_type == 'projects':
                    item_text = f"{evidence.source_item.title} (Project)"
                elif evidence.item_type == 'hobbies':
                    item_text = f"{evidence.source_item.name} (Hobby)"

                # Create the MappingPair
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