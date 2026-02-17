# backend/core/services/scoring.py

from typing import List, Optional
from backend.core.models import JobDescription, CV, JobFitStats, Application, Mapping, ForensicAnalysis
from backend.core.registry import Registry
from backend.core.inferer import MappingInferer
from backend.core.forensics import ForensicCalculator
from backend.core.services.mapping_service import SmartMapper

class ScoringService:
    def __init__(self, registry: Registry, inferer: Optional[MappingInferer] = None):
        self.registry = registry
        # Allow passing an existing inferer to prevent reloading heavy models
        self.inferer = inferer if inferer else MappingInferer() 
        self.forensics = ForensicCalculator()

    # --- SCENARIO A: BROWSING (Score Job against Default CV) ---
    def score_job(self, user_id: str, job_id: str, cv_id: str) -> Optional[JobDescription]:
        """
        Calculates the "Browsing Score" for a job using the User's Default CV.
        Updates the JobDescription cache directly.
        """
        job = self.registry.get_job(job_id, user_id)
        cv = self.registry.get_cv(cv_id, user_id)
        if not job or not cv:
            return None

        # 1. Run Inference & Smart Merge
        mapping = self._get_or_create_smart_mapping(user_id, job, cv)
        
        # 2. Run Forensics
        analysis = self.forensics.calculate(job, mapping)
        
        # 3. Update Job Cache (The "Spotlight" data)
        # We now pull the grade and badges directly from the authoritative calculation
        job.match_score = analysis.stats.overall_match_score
        job.match_grade = analysis.suggested_grade
        job.cached_badges = analysis.suggested_badges
        
        self.registry.update_job(user_id, job.id, job)
        
        return job

    # --- SCENARIO B: APPLIED (Score Application against Attached CV) ---
    def score_application(self, user_id: str, app_id: str) -> Optional[Application]:
        """
        Refreshes the score for a specific APPLICATION using its attached CV.
        Updates the Application cache, NOT the Job cache.
        """
        app = self.registry.get_application(app_id, user_id)
        if not app: return None

        # Fetch context based on the Application's specific links
        job = self.registry.get_job(app.job_id, user_id)
        cv = self.registry.get_cv(app.base_cv_id, user_id)
        
        if not job or not cv: return None

        # 1. Run Inference & Smart Merge
        mapping = self._get_or_create_smart_mapping(user_id, job, cv, existing_mapping_id=app.mapping_id)
        
        # 2. Run Forensics
        analysis = self.forensics.calculate(job, mapping)
        
        # 3. Update APPLICATION Cache
        self.update_application_from_analysis(user_id, app, analysis)
        
        return app

    # --- HELPER: CENTRALIZED CACHE UPDATE ---
    
    def update_application_from_analysis(self, user_id: str, app: Application, analysis: ForensicAnalysis):
        """
        Updates the Application record with the authoritative scores, grades, and badges
        returned by the Forensic Analysis.
        """
        # Only update if something changed or badges are missing
        if (app.match_score != analysis.stats.overall_match_score) or (not app.cached_badges):
            app.match_score = analysis.stats.overall_match_score
            app.match_grade = analysis.suggested_grade
            app.cached_badges = analysis.suggested_badges
            
            self.registry.update_application(user_id, app.id, app)

    # --- CORE LOGIC ---

    def _get_or_create_smart_mapping(self, user_id: str, job: JobDescription, cv: CV, existing_mapping_id: str = None) -> Mapping:
        """
        Runs Active Inference (Fresh Look) and merges it with existing User History (Smart Mapping).
        """
        # A. Active Inference (Fresh AI Opinion)
        fresh_pairs = self.inferer.infer_mappings(job, cv, min_score=0.20)

        print(f"[ScoringService] Fresh inference generated {len(fresh_pairs)} pairs for Job {job.id} and CV {cv.id}.")

        # B. Fetch Existing State
        existing_mapping = None
        
        if existing_mapping_id:
            existing_mapping = self.registry.get_mapping(existing_mapping_id, user_id)
        else:
            # Try to find one loosely if ID not provided
            all_maps = self.registry.all_mappings(user_id)
            existing_mapping = next((m for m in all_maps if m.job_id == job.id and m.base_cv_id == cv.id), None)

        print(f"[ScoringService] Existing mapping fetched: {existing_mapping.id if existing_mapping else 'None'} for Job {job.id} and CV {cv.id}. Number of existing pairs: {len(existing_mapping.pairs) if existing_mapping else 'N/A'}")

        # C. The Smart Merge
        if existing_mapping:
            # Merge Fresh + User History
            final_pairs = SmartMapper.merge_inference(existing_mapping, fresh_pairs)
            print(f"[ScoringService] Merged pairs count: {len(final_pairs)} for Mapping {existing_mapping.id}." )
            
            # SORT FIX: Ensure strongest matches are first
            final_pairs.sort(key=lambda x: x.strength or 0, reverse=True)
            
            existing_mapping.pairs = final_pairs
            self.registry.save_mapping(user_id, existing_mapping)
            return existing_mapping
        else:
            # Create New
            new_mapping = self.registry.create_mapping(user_id, job.id, cv.id)
            print(f"[ScoringService] No existing mapping. Created new Mapping {new_mapping.id} for Job {job.id} and CV {cv.id}.")
            
            # SORT FIX
            fresh_pairs.sort(key=lambda x: (x.strength or 0), reverse=True)
            
            new_mapping.pairs = fresh_pairs
            print(f"[ScoringService] Assigned {len(fresh_pairs)} fresh pairs to new Mapping {new_mapping.id}.")
            self.registry.save_mapping(user_id, new_mapping)
            return new_mapping