from backend.core.models import MappingPair
from backend.core.utils.inference import MatchScoring, SmartNoteBuilder
import hashlib

class MappingOptimizer:
    
    @staticmethod
    def reject_current_match(pair: MappingPair) -> bool:
        """
        Rejects the current best_match, promotes the next best, 
        and RECALCULATES the pair's strength.
        """
        if not pair.meta or not pair.meta.best_match:
            return False

        # 1. Archive the current champion
        old_champion = pair.meta.best_match
        pair.meta.rejected_matches.append(old_champion)

        # 2. Check the Bench
        pair.meta.supporting_matches.sort(key=lambda x: x.score, reverse=True)

        if not pair.meta.supporting_matches:
            # GAP CREATED: No backups available
            pair.strength = 0.0
            pair.context_item_id = None
            pair.context_item_type = None
            pair.context_item_text = "No valid evidence found."
            pair.meta.summary_note = "User rejected all evidence."
            pair.meta.best_match = None
            return False

        # 3. Promote the new Champion
        new_champion = pair.meta.supporting_matches.pop(0) 
        pair.meta.best_match = new_champion

        # 4. Instant Recalculation
        active_candidates = [new_champion] + pair.meta.supporting_matches
        pair.strength = MatchScoring.aggregate_pair_strength(active_candidates)

        # 5. Update Context Links
        MappingOptimizer._update_pair_context(pair, new_champion)
        
        return True

    @staticmethod
    def promote_alternative(pair: MappingPair, alternative_id: str) -> bool:
        """
        Promotes a specific supporting match to be the Best Match.
        The old Best Match is moved to the supporting list (bench).
        """
        if not pair.meta or not pair.meta.supporting_matches:
            return False

        # 1. Find the candidate by regenerating the hash
        candidate_to_promote = None
        promo_index = -1
        
        for idx, cand in enumerate(pair.meta.supporting_matches):
            unique_str = f"{cand.segment_text}_{cand.score}"
            cand_hash = hashlib.md5(unique_str.encode()).hexdigest()
            
            if cand_hash == alternative_id:
                candidate_to_promote = cand
                promo_index = idx
                break
        
        if not candidate_to_promote:
            return False

        # 2. Swap Positions
        current_best = pair.meta.best_match
        
        # Remove winner from bench
        pair.meta.supporting_matches.pop(promo_index)
        
        # Add loser to bench
        if current_best:
            pair.meta.supporting_matches.append(current_best)
            
        # Set new best
        pair.meta.best_match = candidate_to_promote
        
        # 3. Boost Score (User intervention implies 100% confidence)
        candidate_to_promote.score = 1.0 
        pair.strength = 1.0
        
        # 4. Update Context
        MappingOptimizer._update_pair_context(pair, candidate_to_promote)
        
        return True

    @staticmethod
    def approve_current_match(pair: MappingPair) -> bool:
        """
        User explicitly verifies the current match. 
        Sets confidence to 1.0 to lock it in visually.
        """
        if not pair.meta or not pair.meta.best_match:
            return False
            
        pair.meta.best_match.score = 1.0
        pair.strength = 1.0 # Force max strength
        
        # Refresh the note just in case
        MappingOptimizer._update_pair_context(pair, pair.meta.best_match)
        return True

    @staticmethod
    def _update_pair_context(pair, champion):
        """Helper to update the pair's pointers based on the winning candidate."""
        if champion.lineage:
            root = champion.lineage[0]
            pair.context_item_id = root.id
            pair.context_item_type = root.type
            pair.context_item_text = root.name
            
        pair.meta.summary_note = SmartNoteBuilder.build(
            text=champion.segment_text,
            score=champion.score,
            paths=[champion.lineage]
        )