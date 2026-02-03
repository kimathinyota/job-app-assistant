from backend.core.models import MappingPair
from backend.core.utils.inference import MatchScoring, SmartNoteBuilder

class MappingOptimizer:
    
    @staticmethod
    def reject_current_match(pair: MappingPair) -> bool:
        """
        Rejects the current best_match, promotes the next best, 
        and RECALCULATES the pair's strength.
        Returns True if a replacement was found, False if pair is now empty (Gap).
        """
        if not pair.meta or not pair.meta.best_match:
            return False

        # 1. Archive the current champion
        old_champion = pair.meta.best_match
        pair.meta.rejected_matches.append(old_champion)

        # 2. Check the Bench
        # Ensure supporting matches are sorted by score descending
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
        # We calculate strength based on the new active team (Champion + Bench)
        active_candidates = [new_champion] + pair.meta.supporting_matches
        pair.strength = MatchScoring.aggregate_pair_strength(active_candidates)

        # 5. Update Context Links (For UI Navigation)
        if new_champion.lineage:
            root_context = new_champion.lineage[0] 
            pair.context_item_id = root_context.id
            pair.context_item_type = root_context.type
            pair.context_item_text = root_context.name
        
        # 6. Regenerate Smart Note
        pair.meta.summary_note = SmartNoteBuilder.build(
            text=new_champion.segment_text,
            score=new_champion.score,
            paths=[new_champion.lineage] 
        )
        
        return True