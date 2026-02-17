# backend/core/services/scoring.py

from typing import List, Optional
from backend.core.models import JobDescription, CV, JobFitStats, Application, Mapping
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
        self._update_job_cache(user_id, job, analysis.stats)
        
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
        # This ensures the "Applied" card shows the score for the CV used to apply,
        # which might be different from the default CV.
        app.match_score = analysis.stats.overall_match_score
        app.match_grade = self._calculate_grade(analysis.stats.overall_match_score)
        app.cached_badges = self._generate_badges(analysis.stats, job)
        
        self.registry.update_application(user_id, app.id, app)
        
        return app

    # --- CORE LOGIC ---

    def _get_or_create_smart_mapping(self, user_id: str, job: JobDescription, cv: CV, existing_mapping_id: str = None) -> Mapping:
        """
        Runs Active Inference (Fresh Look) and merges it with existing User History (Smart Mapping).
        """
        # A. Active Inference (Fresh AI Opinion)
        fresh_pairs = self.inferer.infer_mappings(job, cv, min_score=0.20)

        # B. Fetch Existing State
        existing_mapping = None
        
        if existing_mapping_id:
            existing_mapping = self.registry.get_mapping(existing_mapping_id, user_id)
        else:
            # Try to find one loosely if ID not provided
            all_maps = self.registry.all_mappings(user_id)
            existing_mapping = next((m for m in all_maps if m.job_id == job.id and m.base_cv_id == cv.id), None)

        # C. The Smart Merge
        if existing_mapping:
            # Merge Fresh + User History
            final_pairs = SmartMapper.merge_inference(existing_mapping, fresh_pairs)
            existing_mapping.pairs = final_pairs
            self.registry.save_mapping(user_id, existing_mapping)
            return existing_mapping
        else:
            # Create New
            new_mapping = self.registry.create_mapping(user_id, job.id, cv.id)
            new_mapping.pairs = fresh_pairs
            self.registry.save_mapping(user_id, new_mapping)
            return new_mapping

    def _update_job_cache(self, user_id: str, job: JobDescription, stats: JobFitStats):
        """Writes score and badges to the Job record."""
        job.match_score = stats.overall_match_score
        job.match_grade = self._calculate_grade(stats.overall_match_score)
        job.cached_badges = self._generate_badges(stats, job)
        self.registry.update_job(user_id, job.id, job)

    def _calculate_grade(self, score: float) -> str:
        if score >= 85: return "A"
        if score >= 65: return "B"
        if score >= 40: return "C"
        return "D"

    def _generate_badges(self, stats: JobFitStats, job: JobDescription) -> List[str]:
        """
        Generates rich, descriptive tags for the card.
        This provides the "Better Tags" for the frontend.
        """
        badges = []
        
        # 1. Critical Flags (The Dealbreakers)
        if stats.critical_gaps_count > 0:
            badges.append("Missing Critical Skills")
        
        # 2. Score Highlights
        if stats.overall_match_score >= 90:
            badges.append("Top Match 🚀")
        elif stats.overall_match_score >= 75:
            badges.append("Strong Fit")
        elif stats.overall_match_score < 30:
            badges.append("Poor Fit")

        # 3. Shape of Skills (The "Why")
        if stats.coverage_pct > 80 and stats.overall_match_score < 60:
            badges.append("Broad but Weak") # Lots of keywords, low proof
        
        if stats.coverage_pct < 50 and stats.overall_match_score > 70:
            badges.append("Niche Specialist") # Few matches, but very strong evidence

        # 4. Job Metadata Tags (Easy wins for visuals)
        if job.location and "remote" in job.location.lower():
            badges.append("Remote")
            
        if job.salary_range:
            # Simple heuristic to flag salary presence
            clean_sal = job.salary_range.lower()
            if "k" in clean_sal or "£" in clean_sal or "$" in clean_sal: 
                badges.append("Salary Listed")

        return badges