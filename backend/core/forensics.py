# backend/core/forensics.py

from typing import List, Dict, Any, Optional
from backend.core.models import (
    JobDescription, Mapping, MappingPair,
    ForensicAnalysis, JobFitStats, ForensicItem
)
from collections import defaultdict


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
        
        evidence_counts = defaultdict(int)
        evidence_ids_by_source = defaultdict(list)
        
        # Scoring Accumulators
        total_possible_weight = 0.0
        earned_weighted_score = 0.0
        sum_quality_of_matches = 0.0
        
        total_scorable_reqs = 0
        met_reqs = 0
        missing_critical_ids = []

        # 2. Filter Scorable Features
        ignored_types = ['benefit', 'other', 'salary', 'employer_mission', 'employer_culture', 'role_value']
        scorable_features = [f for f in job.features if f.type not in ignored_types]

        # 3. Create Fast Lookup for Matches
        pair_map = {p.feature_id: p for p in mapping.pairs}

        # 4. The Analysis Loop
        for feature in scorable_features:
            total_scorable_reqs += 1
            
            # A. Get Match Data
            pair = pair_map.get(feature.id)
            match_confidence = pair.strength if pair else 0.0
            
            # B. Determine Importance & Weight
            weight = self._get_weight(feature.type)
            importance_label = self._get_importance_label(weight)
            total_possible_weight += weight

            # C. Determine Status
            status = "missing"
            if match_confidence >= 0.75: status = "verified"
            elif match_confidence >= 0.35: status = "pending"
            
            # D. Update Metrics
            if status != "missing":
                met_reqs += 1
                earned_weighted_score += (match_confidence * weight)
                sum_quality_of_matches += match_confidence
                
                # Authority Classification
                authority_bucket = self._classify_authority(pair)
                evidence_counts[authority_bucket] += 1
                evidence_ids_by_source[authority_bucket].append(feature.id)
            else:
                authority_bucket = "Missing"

            # E. Gap Analysis
            if status == "missing" and weight >= 1.25:
                missing_critical_ids.append(feature.id)

            # F. Build the Forensic Item (Updated for UI Interactivity)
            best_match_excerpt = None
            summary_note = None
            structured_lineage = [] # <--- NEW: Container for the clickable chips
            
            if pair and pair.meta:
                # 1. Get the text summary (e.g. "Excerpt... (Exp > Proj)")
                summary_note = pair.meta.summary_note 
                
                if pair.meta.best_match:
                    best_match_excerpt = pair.meta.best_match.segment_text
                    # 2. COPY THE STRUCTURED LINEAGE OBJECTS
                    # This allows the frontend to iterate and create clickable links
                    structured_lineage = pair.meta.best_match.lineage

            item = ForensicItem(
                requirement_id=feature.id,
                requirement_text=feature.description,
                requirement_type=feature.type,
                importance=importance_label,
                status=status,
                
                # Links
                best_match_id=pair.id if pair else None,
                cv_item_id=pair.context_item_id if pair else None,
                cv_item_type=pair.context_item_type if pair else None,
                
                # --- INTERACTIVITY SUPPORT ---
                lineage=structured_lineage, # Passes the list of objects with IDs
                
                # Display
                best_match_text=pair.context_item_text if pair else None,
                best_match_excerpt=best_match_excerpt,
                best_match_confidence=match_confidence,
                match_summary=summary_note, # Renamed field (prev. lineage_text)
                
                # Filter
                authority_bucket=authority_bucket
            )
            
            groups[importance_label].append(item)

        # 5. Final Calculations
        overall_score = 0.0
        if total_possible_weight > 0:
            overall_score = (earned_weighted_score / total_possible_weight) * 100
            
        coverage_pct = 0.0
        if total_scorable_reqs > 0:
            coverage_pct = (met_reqs / total_scorable_reqs) * 100
            
        average_quality = 0.0
        if met_reqs > 0:
            average_quality = (sum_quality_of_matches / met_reqs) * 100

        stats = JobFitStats(
            overall_match_score=round(overall_score, 1),
            coverage_pct=round(coverage_pct, 1),
            average_quality=round(average_quality, 1),
            
            total_reqs=total_scorable_reqs,
            met_reqs=met_reqs,
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
        if "hobby" in root: return "Personal"
        if "project" in root: return "Project"
        
        if pair.meta and pair.meta.best_match and pair.meta.best_match.lineage:
            types = [x.type.lower() for x in pair.meta.best_match.lineage]
            if "experience" in types: return "Professional"
            if "education" in types: return "Academic"
            if "hobby" in types: return "Personal"
            
        return "Other"