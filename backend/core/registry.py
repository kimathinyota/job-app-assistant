from typing import Optional, List, Dict
from backend.core.database import TinyDBManager
from tinydb import where
from backend.core.models import *


class Registry:
    """Central object-relational manager for all entities."""

    def __init__(self, db_path: str = "./backend/data/db.json"):
        self.db = TinyDBManager(db_path)

    # ---- Generic Helpers ----

    def _insert(self, table: str, obj: BaseEntity):
        """Insert or replace entity in TinyDB."""
        self.db.upsert(table, obj)
        return obj

    def _get(self, table: str, cls, obj_id: str):
        """Fetch entity by ID."""
        data = self.db.get(table, obj_id)
        return cls(**data) if data else None

    def _all(self, table: str, cls):
        """Return all entities of a given type."""
        return [cls(**d) for d in self.db.all(table)]

    def _update(self, table: str, obj: BaseEntity):
        """Update an existing entity."""
        self.db.update(table, obj)

    # ---- Jobs ----
    def create_job(self, title: str, company: str, notes: Optional[str] = None):
        job = JobDescription.create(title=title, company=company, notes=notes)
        return self._insert("jobs", job)

    def add_job_feature(self, job_id: str, description: str, type: str = "requirement"):
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")
        feature = JobDescriptionFeature.create(description=description, type=type)
        job.add_feature(feature)
        self._update("jobs", job)
        return feature

    def get_job(self, job_id: str):
        return self._get("jobs", JobDescription, job_id)

    def all_jobs(self):
        return self._all("jobs", JobDescription)

    # ---- CVs ----
    def create_cv(self, name: str, summary: Optional[str] = None):
        cv = CV.create(name=name, summary=summary)
        return self._insert("cvs", cv)

    def get_cv(self, cv_id: str):
        return self._get("cvs", CV, cv_id)

    def all_cvs(self):
        return self._all("cvs", CV)

    # ---- Mappings ----
    def create_mapping(self, job_id: str, base_cv_id: str):
        mapping = Mapping.create(job_id=job_id, base_cv_id=base_cv_id)
        return self._insert("mappings", mapping)

    def add_mapping_pair(self, mapping_id: str, feature, experience, annotation: Optional[str] = None):
        mapping = self.get_mapping(mapping_id)
        if not mapping:
            raise ValueError("Mapping not found")
        pair = MappingPair.create(feature, experience, annotation)
        mapping.add_pair(pair)
        self._update("mappings", mapping)
        return pair

    def get_mapping(self, mapping_id: str):
        return self._get("mappings", Mapping, mapping_id)

    def all_mappings(self):
        return self._all("mappings", Mapping)

    # ---- Applications ----
    def create_application(
        self, job_id: str, base_cv_id: str,
        mapping_id: Optional[str] = None, derived_cv_id: Optional[str] = None
    ):
        app = Application.create(job_id, base_cv_id, mapping_id, derived_cv_id)
        return self._insert("applications", app)

    def get_application(self, app_id: str):
        return self._get("applications", Application, app_id)

    def all_applications(self):
        return self._all("applications", Application)

    # ---- Cover Letters ----
    def create_cover_letter(self, job_id: str, base_cv_id: str, mapping_id: str):
        cover = CoverLetter.create(job_id, base_cv_id, mapping_id)
        return self._insert("coverletters", cover)

    def get_cover_letter(self, cover_id: str):
        return self._get("coverletters", CoverLetter, cover_id)

    # ---- Interviews ----
    def create_interview(self, application_id: str):
        interview = Interview.create(application_id=application_id)
        return self._insert("interviews", interview)

    def add_interview_stage(self, interview_id: str, name: str, description: Optional[str] = None):
        interview = self.get_interview(interview_id)
        if not interview:
            raise ValueError("Interview not found")
        stage = InterviewStage.create(name, description)
        interview.add_stage(stage)
        self._update("interviews", interview)
        return stage

    def add_interview_question(self, interview_id: str, stage_name: str, question: str, answer: Optional[str] = None):
        interview = self.get_interview(interview_id)
        if not interview:
            raise ValueError("Interview not found")
        stage = next((s for s in interview.stages if s.name == stage_name), None)
        if not stage:
            raise ValueError("Stage not found")
        q = InterviewQuestion.create(question, answer)
        stage.add_question(q)
        self._update("interviews", interview)
        return q

    def get_interview(self, interview_id: str):
        return self._get("interviews", Interview, interview_id)

    # ---- Work Items ----
    def create_work_item(
        self,
        title: str,
        description: Optional[str] = None,
        work_type: str = "task",
        related_application_id: Optional[str] = None,
        related_goal_id: Optional[str] = None
    ):
        work = WorkItem.create(
            title=title,
            description=description,
            work_type=work_type,
            related_application_id=related_application_id,
            related_goal_id=related_goal_id
        )
        return self._insert("work_items", work)

    def get_work_item(self, work_id: str):
        return self._get("work_items", WorkItem, work_id)

    def all_work_items(self):
        return self._all("work_items", WorkItem)

    # ---- Goals ----
    def create_goal(self, title: str, description: Optional[str] = None, metric: Optional[str] = None):
        goal = Goal.create(title=title, description=description, metric=metric)
        return self._insert("goals", goal)

    def get_goal(self, goal_id: str):
        return self._get("goals", Goal, goal_id)

    def all_goals(self):
        return self._all("goals", Goal)

    def add_work_to_goal(self, goal_id: str, work_id: str):
        goal = self.get_goal(goal_id)
        work = self.get_work_item(work_id)
        if not goal or not work:
            raise ValueError("Goal or WorkItem not found")
        goal.add_work_item(work)
        self._update("goals", goal)
        self._update("work_items", work)
        return goal

    # ---- Relationship Helpers ----
    def get_job_for_application(self, app_id: str):
        app = self.get_application(app_id)
        return self.get_job(app.job_id) if app else None

    def get_mapping_for_application(self, app_id: str):
        app = self.get_application(app_id)
        return self.get_mapping(app.mapping_id) if app and app.mapping_id else None

    def get_cv_for_application(self, app_id: str):
        app = self.get_application(app_id)
        return self.get_cv(app.base_cv_id) if app else None
