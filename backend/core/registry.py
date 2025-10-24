from typing import Optional, List, Dict
from backend.core.database import TinyDBManager
from backend.core.models import BaseEntity, CV, JobDescription, JobDescriptionFeature, Mapping, MappingPair, Experience, DerivedCV, CoverLetter, Application, Interview, InterviewStage, InterviewQuestion, WorkItem



class Registry:
    """Central data access layer managing TinyDB and all model relationships."""

    def __init__(self, db_path: str = "./backend/data/db.json"):
        self.db = TinyDBManager(db_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _insert(self, table: str, obj: BaseEntity):
        self.db.insert(table, obj)
        return obj

    def _get(self, table: str, cls, obj_id: str):
        data = self.db.get(table, obj_id)
        return cls(**data) if data else None

    def _all(self, table: str, cls):
        return [cls(**d) for d in self.db.all(table)]

    # ------------------------------------------------------------------
    # CV Management
    # ------------------------------------------------------------------
    def create_cv(self, name: str, summary: Optional[str] = None) -> CV:
        cv = CV.create(name=name, summary=summary)
        return self._insert("cvs", cv)

    def get_cv(self, cv_id: str) -> Optional[CV]:
        return self._get("cvs", CV, cv_id)

    def all_cvs(self) -> List[CV]:
        return self._all("cvs", CV)

    # ------------------------------------------------------------------
    # Job Management
    # ------------------------------------------------------------------
    def create_job(self, title: str, company: str, notes: Optional[str] = None) -> JobDescription:
        job = JobDescription.create(title=title, company=company, notes=notes)
        return self._insert("jobs", job)

    def add_job_feature(self, job_id: str, description: str, type: str = "requirement"):
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")
        feature = job.add_feature(description=description, type=type)
        self.db.insert("jobs", job)  # persist updated job
        return feature

    def get_job(self, job_id: str) -> Optional[JobDescription]:
        return self._get("jobs", JobDescription, job_id)

    def all_jobs(self) -> List[JobDescription]:
        return self._all("jobs", JobDescription)

    # ------------------------------------------------------------------
    # Mapping Management
    # ------------------------------------------------------------------
    def create_mapping(self, job_id: str, base_cv_id: str) -> Mapping:
        mapping = Mapping.create(job_id=job_id, base_cv_id=base_cv_id)
        return self._insert("mappings", mapping)

    def add_mapping_pair(
        self,
        mapping_id: str,
        feature: JobDescriptionFeature,
        experience: Experience,
        annotation: Optional[str] = None,
    ) -> MappingPair:
        mapping = self.get_mapping(mapping_id)
        if not mapping:
            raise ValueError("Mapping not found")
        pair = mapping.add_pair(feature, experience, annotation)
        self.db.insert("mappings", mapping)
        return pair

    def get_mapping(self, mapping_id: str) -> Optional[Mapping]:
        return self._get("mappings", Mapping, mapping_id)

    # ------------------------------------------------------------------
    # Derived CV
    # ------------------------------------------------------------------
    def create_derived_cv(self, base_cv: CV, job_id: str, mapping: Mapping) -> DerivedCV:
        derived = DerivedCV.from_mapping(base_cv, job_id, mapping)
        return self._insert("derived_cvs", derived)

    def get_derived_cv(self, derived_id: str) -> Optional[DerivedCV]:
        return self._get("derived_cvs", DerivedCV, derived_id)

    # ------------------------------------------------------------------
    # Cover Letter
    # ------------------------------------------------------------------
    def create_cover_letter(self, job_id: str, base_cv_id: str, mapping_id: str) -> CoverLetter:
        cover = CoverLetter.create(job_id=job_id, base_cv_id=base_cv_id, mapping_id=mapping_id)
        return self._insert("coverletters", cover)

    def get_cover_letter(self, cover_id: str) -> Optional[CoverLetter]:
        return self._get("coverletters", CoverLetter, cover_id)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def create_application(
        self,
        job_id: str,
        base_cv_id: str,
        mapping_id: Optional[str] = None,
        derived_cv_id: Optional[str] = None,
        cover_letter_id: Optional[str] = None,
    ) -> Application:
        app = Application.create(
            job_id=job_id,
            base_cv_id=base_cv_id,
            mapping_id=mapping_id,
            derived_cv_id=derived_cv_id,
            cover_letter_id=cover_letter_id,
        )
        return self._insert("applications", app)

    def get_application(self, app_id: str) -> Optional[Application]:
        return self._get("applications", Application, app_id)

    def all_applications(self) -> List[Application]:
        return self._all("applications", Application)

    # ------------------------------------------------------------------
    # Interviews
    # ------------------------------------------------------------------
    def create_interview(self, application_id: str) -> Interview:
        interview = Interview.create(application_id=application_id)
        return self._insert("interviews", interview)

    def add_interview_stage(self, interview_id: str, name: str, description: Optional[str] = None) -> InterviewStage:
        interview = self.get_interview(interview_id)
        if not interview:
            raise ValueError("Interview not found")
        stage = interview.add_stage(name=name, description=description)
        self.db.insert("interviews", interview)
        return stage

    def add_interview_question(
        self,
        interview_id: str,
        stage_name: str,
        question: str,
        answer: Optional[str] = None,
    ) -> InterviewQuestion:
        interview = self.get_interview(interview_id)
        if not interview:
            raise ValueError("Interview not found")

        stage = next((s for s in interview.stages if s.name == stage_name), None)
        if not stage:
            raise ValueError("Stage not found")

        q = InterviewQuestion.create(question=question, answer=answer, stage=stage_name)
        stage.questions.append(q)
        self.db.insert("interviews", interview)
        return q

    def get_interview(self, interview_id: str) -> Optional[Interview]:
        return self._get("interviews", Interview, interview_id)
    
    # Inside Registry class

    # ---- Work Items ----
    def create_work_item(self, **kwargs):
        from backend.core.models import WorkItem
        work = WorkItem.create(**kwargs)
        self._add("work_items", work)
        return work

    def all_work_items(self):
        return [WorkItem(**item) for item in self.db.table("work_items").all()]

    def get_work_item(self, work_id: str):
        table = self.db.table("work_items")
        item = table.get(where("id") == work_id)
        return WorkItem(**item) if item else None

    def mark_work_item_completed(self, work_id: str, reflection: Optional[str] = None):
        work = self.get_work_item(work_id)
        if not work:
            return {"error": "WorkItem not found"}
        work.mark_completed(reflection)
        self._update("work_items", work)
        return work

