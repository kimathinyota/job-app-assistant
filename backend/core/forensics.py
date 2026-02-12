# backend/core/forensics.py

from typing import List, Dict, Any, Optional
from backend.core.models import (
    JobDescription, Mapping, MappingPair,
    ForensicAnalysis, JobFitStats, ForensicItem, ForensicAlternative
)
from collections import defaultdict
import hashlib

class ForensicCalculator:
    
    # Weights determine "Importance" and score impact
    FEATURE_WEIGHTS = {
        # Core Constraints
        "experience": 1.5, 
        "qualification": 1.5, 
        "requirement": 1.5,
        "hard_skill": 1.25,
        "tool": 1.25,
        
        # Standard
        "responsibility": 1.0,
        
        # Soft / Bonus
        "soft_skill": 0.5, 
        "nice_to_have": 0.5, 
        "benefit": 0.0, 
        "other": 0.0
    }

    def calculate(self, job: JobDescription, mapping: Mapping) -> ForensicAnalysis:
        # 1. Initialize Buckets
        groups = {
            "Critical": [], 
            "High": [], 
            "Standard": [], 
            "Bonus": []
        }
        
        # Scoring Accumulators
        total_possible_weight = 0.0
        earned_score_curved = 0.0  # For the "Strict" Score (Quality)
        earned_score_linear = 0.0  # For the "Coverage" Score (Quantity)
        
        # Counters
        strong_matches = 0
        weak_matches = 0
        missing_critical_ids = []
        
        # Tracking for Metadata
        evidence_counts = defaultdict(int)
        evidence_ids_by_source = defaultdict(list)

        # --- FIX 1: DEDUPLICATION (The "Champion" Logic) ---
        # mapping.pairs is sorted by strength DESC (Best -> Worst) by the inferer.
        # We must keep the FIRST pair we see for each feature_id.
        pair_map = {}
        for p in mapping.pairs:
            if p.feature_id not in pair_map:
                pair_map[p.feature_id] = p
            # Else: Ignore subsequent (weaker) pairs for scoring purposes

        # 2. Iterate Job Requirements
        for feature in job.features:
            weight = self._get_weight(feature.type)
            bucket = self._get_importance_label(weight)
            
            pair = pair_map.get(feature.id)
            
            # --- SCORING LOGIC ---
            # Filter absolute noise (< 15%)
            if pair and (pair.strength or 0.0) > 0.15: 
                raw_strength = pair.strength
                
                # A. Traffic Light Status
                if raw_strength >= 0.7:
                    status = "verified" # Green
                    strong_matches += 1
                elif raw_strength >= 0.32: # Lowered threshold for semantic/implied matches
                    status = "pending"  # Yellow (Weak/Implied)
                    weak_matches += 1
                else:
                    status = "missing"  # Red (Too weak to count as a "hit")
                    
                # B. The Confidence Curve (Non-Linear Scoring)
                # Square the strength to punish weak matches.
                # 0.9 match -> 0.81 points
                # 0.4 match -> 0.16 points (Noise suppression)
                curved_strength = raw_strength ** 2
                
                # Only add points if it's not a "Red" status match
                if status != "missing":
                    earned_score_linear += (raw_strength * weight)
                    earned_score_curved += (curved_strength * weight)
                    
                    # Track Evidence Source
                    auth_bucket = self._classify_authority(pair)
                    evidence_counts[auth_bucket] += 1
                    evidence_ids_by_source[auth_bucket].append(feature.id)

                # Create Item for UI
                item = ForensicItem(
                    requirement_id=feature.id,
                    requirement_text=feature.description,
                    requirement_type=feature.type,
                    importance=bucket,
                    status=status,
                    
                    # Navigation Data
                    best_match_id=pair.context_item_id, # e.g. exp_123
                    cv_item_id=pair.context_item_id,
                    cv_item_type=pair.context_item_type,
                    
                    # Display Data
                    best_match_text=pair.context_item_text,
                    best_match_excerpt=pair.annotation, # The snippet found
                    best_match_confidence=raw_strength,
                    match_summary=self._generate_summary(pair, raw_strength),
                    
                    authority_bucket=self._classify_authority(pair),
                    
                    # Pass lineage for clickable chips (Exp -> Project -> Skill)
                    lineage=pair.meta.best_match.lineage if (pair.meta and pair.meta.best_match) else []
                )
                
                # Add Alternatives (Supporting Evidence)
                if pair.meta and pair.meta.supporting_matches:
                    for sm in pair.meta.supporting_matches:
                        alt = ForensicAlternative(
                            id=hashlib.md5(sm.segment_text.encode()).hexdigest(),
                            match_text=sm.segment_text,
                            score=sm.score,
                            source_type=pair.context_item_type or "unknown",
                            source_name=pair.context_item_text or "Unknown",
                            lineage=sm.lineage,
                            cv_item_id=pair.context_item_id
                        )
                        item.alternatives.append(alt)

                groups[bucket].append(item)

            else:
                # --- MISSING LOGIC ---
                if bucket == "Critical":
                    missing_critical_ids.append(feature.id)
                
                groups[bucket].append(ForensicItem(
                    requirement_id=feature.id,
                    requirement_text=feature.description,
                    requirement_type=feature.type,
                    importance=bucket,
                    status="missing",
                    match_summary="No evidence found in CV."
                ))
            
            # Always track total weight for the denominator
            total_possible_weight += weight

        # 3. Final Calculations
        if total_possible_weight > 0:
            # The "Strict" Score (Use this for sorting/spotlight)
            overall_score = (earned_score_curved / total_possible_weight) * 100
            
            # The "Potential" Score (How much ground did we cover?)
            coverage_score = (earned_score_linear / total_possible_weight) * 100
        else:
            overall_score = 0.0
            coverage_score = 0.0

        # Calculate Average Quality (of matches only)
        total_matches = strong_matches + weak_matches
        avg_quality = 0.0
        if total_matches > 0:
             # Average of the raw strengths found
             avg_quality = (sum([p.strength for p in pair_map.values() if (p.strength or 0) > 0.15]) / total_matches) * 100

        stats = JobFitStats(
            overall_match_score=round(overall_score, 1),
            coverage_pct=round(coverage_score, 1),
            average_quality=round(avg_quality, 1),
            total_reqs=len(job.features),
            met_reqs=total_matches,
            critical_gaps_count=len(missing_critical_ids),
            missing_critical_ids=missing_critical_ids,
            evidence_sources=dict(evidence_counts),
            evidence_ids_by_source=dict(evidence_ids_by_source)
        )

        return ForensicAnalysis(
            application_id=None,
            stats=stats,
            groups=groups
        )

    # --- HELPERS ---

    def _get_weight(self, f_type: str) -> float:
        return self.FEATURE_WEIGHTS.get(f_type.lower(), 0.5)

    def _get_importance_label(self, weight: float) -> str:
        if weight >= 1.5: return "Critical"
        if weight >= 1.25: return "High"
        if weight >= 1.0: return "Standard"
        return "Bonus"

    def _classify_authority(self, pair: MappingPair) -> str:
        if not pair or not pair.context_item_type:
            return "Missing"
            
        root = pair.context_item_type.lower()
        if "experience" in root: return "Professional"
        if "education" in root: return "Academic"
        if "hobby" in root: return "Other" # Changed from Personal to Other to match test output
        if "project" in root: return "Professional" # Projects are usually professional evidence
        if "skill" in root: return "Other" # Direct skill list is less authoritative than experience
        
        return "Other"

    def _generate_summary(self, pair: MappingPair, score: float) -> str:
        """Generates a human-readable reason for the match."""
        base = f"Excerpt: \"{pair.annotation}\" ({pair.context_item_type.title()}: {pair.context_item_text})"
        confidence = f"[Confidence: {int(score * 100)}%]"
        
        if score >= 0.7:
            return f"{base} {confidence} - Strong Match"
        elif score >= 0.32:
            return f"{base} {confidence}"
        else:
            return f"{base} {confidence} - Weak Match"