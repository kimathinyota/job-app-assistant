
from typing import List, Tuple, Any, Union


class MatchScoring:
    # Central configuration for domain bonuses
    TYPE_BONUS_MATRIX = {
        ("hard_skill", "skill"): 0.20, 
        ("soft_skill", "skill"): 0.15,
        ("responsibility", "achievement"): 0.15, 
        ("hard_skill", "achievement"): 0.15,
        ("responsibility", "experience"): 0.10, 
        ("hard_skill", "project"): 0.10,
        ("qualification", "education"): 0.15, 
        ("benefit", "hobby"): 0.05
    }

    @staticmethod
    def calculate_strength(
        req_type: str, 
        evidence_type: str, 
        tfidf_sim: float, 
        emb_sim: float = 0.0
    ) -> float:
        """
        Calculates the score for a SINGLE candidate match (Evidence Atom).
        """
        # 1. Hybrid Mixing Logic
        if tfidf_sim > 0.7:
            base_score = tfidf_sim
        else:
            # If embedding exists (>0), weight it 70%, otherwise fallback to tfidf
            base_score = (0.3 * tfidf_sim) + (0.7 * emb_sim) if emb_sim > 0 else tfidf_sim

        # 2. Heuristic Penalties (The "Gatekeeper")
        if req_type in ["hard_skill", "qualification", "tool"]:
            # Strict penalty for technical mismatches
            if tfidf_sim < 0.1: 
                return 0.0 
        elif tfidf_sim < 0.05:
            # Soft penalty for other types if keyword match is abysmal
            base_score = base_score * 0.4 

        # 3. Type Bonuses
        type_bonus = 0.0
        if base_score > 0.35:
             type_bonus = MatchScoring.TYPE_BONUS_MATRIX.get((req_type, evidence_type), 0.0)
        
        final_score = base_score + type_bonus
        
        return min(final_score, 1.0)

    @staticmethod
    def aggregate_pair_strength(candidates: List[Any]) -> float:
        """
        Calculates the total strength of a MappingPair based on ALL its matches.
        Expects a list of objects (MatchCandidates) that have a .score attribute.
        """
        if not candidates:
            return 0.0
            
        scores = [c.score for c in candidates]
        
        # Current Logic: Winner Takes All
        return max(scores)



class SmartNoteBuilder:
    
    @staticmethod
    def _extract_name_type(step: Any) -> tuple:
        """Helper to handle both Dict (Inferer) and Pydantic Object (Optimizer)"""
        if isinstance(step, dict):
            return step.get('name', ''), step.get('type', '')
        return getattr(step, 'name', ''), getattr(step, 'type', '')

    @staticmethod
    def build(text: str, score: float, paths: List[List[Any]]) -> str:
        """
        Generates the standard 'Smart Note' used in the UI.
        
        Args:
            text: The matched segment text.
            score: The match strength (0.0 - 1.0).
            paths: A list of lineage chains. 
                   (Inferer passes all occurrences; Optimizer passes the single new lineage)
        """
        # 1. Process Lineage Paths into Strings
        raw_locations = []
        for path in paths:
            # Format: "Type: Name > Type: Name"
            steps_str = []
            for step in path:
                name, item_type = SmartNoteBuilder._extract_name_type(step)
                steps_str.append(f"{item_type.title()}: {name}")
            raw_locations.append(" > ".join(steps_str))

        # 2. Filter Redundant Locations (e.g. don't show "Exp A" if "Exp A > Desc" exists)
        # Sort by length descending so we keep the most specific ones
        unique_locs = sorted(list(set(raw_locations)), key=len, reverse=True)
        final_locs = []
        for loc in unique_locs:
            # If this loc is already a substring of an accepted loc, skip it
            is_redundant = any(loc in accepted for accepted in final_locs)
            if not is_redundant:
                final_locs.append(loc)

        primary_loc = final_locs[0] if final_locs else "Unknown Context"
        others = final_locs[1:]

        # 3. Format Excerpt
        excerpt = text[:72] + "..." if len(text) > 75 else text
        
        # 4. Construct Note
        note = f"Excerpt: \"{excerpt}\" ({primary_loc})"

        # 5. Add "Also found in"
        if others:
            extras = "; ".join(others[:2])
            if len(others) > 2:
                extras += f"; +{len(others)-2} more"
            note += f" [Also found in: {extras}]"

        # 6. Add Confidence
        confidence = int(score * 100)
        note += f" [Confidence: {confidence}%]"
        
        if score > 0.85:
            note += " - Strong Match"

        return note