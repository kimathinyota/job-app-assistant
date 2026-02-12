
from typing import List, Tuple, Any, Union


from typing import List, Any

class MatchScoring:
    # Central configuration for domain bonuses
    # (Kept identical to your original, as these are good heuristics)
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
        Calculates the score for a SINGLE candidate match.
        Implements 'Synonym Rescue' and 'Qualification Strictness'.
        """
        # 1. Hybrid Mixing Logic
        if tfidf_sim > 0.7:
            # If exact keyword match is massive, trust it 100%
            base_score = tfidf_sim
        else:
            # Shift weight to Semantics (0.8) over Keywords (0.2)
            # This captures "implied" skills much better.
            if emb_sim > 0:
                base_score = (0.2 * tfidf_sim) + (0.8 * emb_sim)
            else:
                base_score = tfidf_sim

        # 2. Advanced Heuristics (Caps & Rescues)
        
        # CASE A: Qualifications (Strict Mode)
        # We don't want to "hallucinate" a CPA or MBA if the acronym isn't there.
        if req_type == "qualification":
            if tfidf_sim < 0.1:
                # Force "Weak Match" status (Yellow) regardless of semantic similarity
                base_score = min(base_score, 0.55)

        # CASE B: Hard Skills & Tools (Adaptive Mode)
        elif req_type in ["hard_skill", "tool"]:
            if tfidf_sim < 0.1: 
                # Synonym Rescue: If semantics are overwhelming (>0.85),
                # it's likely a direct synonym (e.g. "JS" vs "JavaScript").
                # We allow it to reach "Green" status (0.75) but not perfection.
                if emb_sim > 0.85:
                    base_score = min(base_score, 0.75)
                else:
                    # Standard Cap: Good match, but missing keyword.
                    # Forces "Yellow" status to prompt user review.
                    base_score = min(base_score, 0.65)
        
        # CASE C: General Noise Filter
        # Only punish low keywords if the semantic meaning is ALSO weak.
        elif tfidf_sim < 0.05:
            if emb_sim < 0.31:
                base_score = base_score * 0.5 

        # 3. Type Bonuses
        type_bonus = 0.0
        # Threshold lowered to 0.30 to allow "implied" matches to benefit from context
        if base_score > 0.30:
             type_bonus = MatchScoring.TYPE_BONUS_MATRIX.get((req_type, evidence_type), 0.0)
        
        final_score = base_score + type_bonus
        
        return min(final_score, 1.0)

    @staticmethod
    def aggregate_pair_strength(candidates: List[Any]) -> float:
        """
        Calculates total strength: The Best Match + A 'Depth Bonus'.
        """
        if not candidates:
            return 0.0
            
        # Sort scores descending (Best match first)
        scores = sorted([c.score for c in candidates], reverse=True)
        best_score = scores[0]
        
        # Depth Bonus:
        # Give a 5% boost based on the strength of the *second* best evidence.
        # This rewards candidates who demonstrate the skill multiple times.
        bonus = 0.0
        if len(scores) > 1:
            bonus = scores[1] * 0.05
            
        return min(best_score + bonus, 1.0)


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