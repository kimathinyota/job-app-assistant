# backend/core/models.py

from __future__ import annotations
from typing import List, Dict, Optional, Literal, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


# ---------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------

def gen_id(prefix: str) -> str:
    """Generate consistent unique IDs with type prefixes."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class BaseEntity(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("entity"))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def touch(self):
        self.updated_at = datetime.utcnow()

    @classmethod
    def create(cls, **kwargs):
        obj = cls(**kwargs)
        obj.id = gen_id(cls.__name__.lower())
        return obj


# ---------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------

class Skill(BaseEntity):
    name: str
    category: Literal["technical", "soft", "language", "other"] = "technical"
    level: Optional[str] = None
    importance: Optional[int] = Field(default=None, ge=1, le=5)  # 1–5 scale
    description: Optional[str] = None


class SkillLinkMixin(BaseModel):
    skill_ids: List[str] = Field(default_factory=list)

    def attach_skill(self, skill: Skill):
        if skill.id not in self.skill_ids:
            self.skill_ids.append(skill.id)


class Achievement(BaseEntity, SkillLinkMixin):
    """Global Achievement entity — reusable across contexts."""
    text: str
    context: Optional[str] = None


# ---------------------------------------------------------------------
# Experience, Education, Project, Hobby
# ---------------------------------------------------------------------

class Experience(BaseEntity, SkillLinkMixin):
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    achievement_ids: List[str] = Field(default_factory=list)

    def add_achievement(self, achievement: Achievement):
        if achievement.id not in self.achievement_ids:
            self.achievement_ids.append(achievement.id)
        self.touch()
        return achievement


class Education(BaseEntity, SkillLinkMixin):
    institution: str
    degree: str
    field: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    achievement_ids: List[str] = Field(default_factory=list)

    def add_achievement(self, achievement: Achievement):
        if achievement.id not in self.achievement_ids:
            self.achievement_ids.append(achievement.id)
        self.touch()
        return achievement


class Project(BaseEntity, SkillLinkMixin):
    title: str
    description: str
    related_experience_id: Optional[str] = None
    related_education_id: Optional[str] = None
    achievement_ids: List[str] = Field(default_factory=list)

    def add_achievement(self, achievement: Achievement):
        if achievement.id not in self.achievement_ids:
            self.achievement_ids.append(achievement.id)
        self.touch()
        return achievement


class Hobby(BaseEntity, SkillLinkMixin):
    name: str
    description: Optional[str] = None
    achievement_ids: List[str] = Field(default_factory=list)

    def add_achievement(self, achievement: Achievement):
        if achievement.id not in self.achievement_ids:
            self.achievement_ids.append(achievement.id)
        self.touch()
        return achievement


# ---------------------------------------------------------------------
# Base and Derived CVs
# ---------------------------------------------------------------------

class CV(BaseEntity):
    """Base CV containing all content."""
    name: str
    summary: Optional[str] = None
    contact_info: Dict[str, str] = Field(default_factory=dict)
    experiences: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    hobbies: List[Hobby] = Field(default_factory=list)
    achievements: List[Achievement] = Field(default_factory=list)

    # --- Helpers ---
    def add_skill(self, name: str, category: str = "technical", **kwargs) -> Skill:
        skill = Skill.create(name=name, category=category, **kwargs)
        self.skills.append(skill)
        self.touch()
        return skill

    def add_experience(self, title: str, company: str, **kwargs) -> Experience:
        exp = Experience.create(title=title, company=company, **kwargs)
        self.experiences.append(exp)
        self.touch()
        return exp

    def add_project(self, title: str, description: str, **kwargs) -> Project:
        proj = Project.create(title=title, description=description, **kwargs)
        self.projects.append(proj)
        self.touch()
        return proj

    def add_hobby(self, name: str, description: Optional[str] = None) -> Hobby:
        hobby = Hobby.create(name=name, description=description)
        self.hobbies.append(hobby)
        self.touch()
        return hobby

    def add_achievement(self, text: str, context: Optional[str] = None) -> Achievement:
        ach = Achievement.create(text=text, context=context)
        self.achievements.append(ach)
        self.touch()
        return ach


class DerivedCV(CV):
    """Job-specific tailored CV derived from a base CV."""
    base_cv_id: str
    job_id: str
    mapping_id: str
    application_id: Optional[str] = None
    selected_experience_ids: List[str] = Field(default_factory=list)
    selected_skill_ids: List[str] = Field(default_factory=list)
    selected_project_ids: List[str] = Field(default_factory=list)

    @classmethod
    def from_mapping(cls, base_cv: CV, job_id: str, mapping: Mapping) -> DerivedCV:
        exp_ids = {p.experience_id for p in mapping.pairs if p.experience_id}
        skill_ids = set()
        for exp in base_cv.experiences:
            if exp.id in exp_ids:
                skill_ids.update(exp.skill_ids)

        return cls.create(
            name=f"{base_cv.name} — tailored for {job_id}",
            base_cv_id=base_cv.id,
            job_id=job_id,
            mapping_id=mapping.id,
            experiences=[e for e in base_cv.experiences if e.id in exp_ids],
            skills=[s for s in base_cv.skills if s.id in skill_ids],
            selected_experience_ids=list(exp_ids),
            selected_skill_ids=list(skill_ids),
        )


# ---------------------------------------------------------------------
# Job description & matching
# ---------------------------------------------------------------------

class JobDescriptionFeature(BaseEntity):
    type: Literal[
        "requirement",
        "responsibility",
        "value",
        "nice_to_have",
        "qualification",
        "benefit",
        "other",
    ] = "requirement"
    description: str


class JobDescription(BaseEntity):
    title: str
    company: str
    features: List[JobDescriptionFeature] = Field(default_factory=list)
    notes: Optional[str] = None

    def add_feature(self, description: str, type: str = "requirement") -> JobDescriptionFeature:
        f = JobDescriptionFeature.create(description=description, type=type)
        self.features.append(f)
        self.touch()
        return f


class MappingPair(BaseEntity):
    feature_id: str
    experience_id: Optional[str] = None
    feature_text: Optional[str] = None
    experience_text: Optional[str] = None
    annotation: Optional[str] = None


class Mapping(BaseEntity):
    job_id: str
    base_cv_id: str
    pairs: List[MappingPair] = Field(default_factory=list)

    def add_pair(
        self,
        feature: JobDescriptionFeature,
        experience: Experience,
        annotation: Optional[str] = None,
    ) -> MappingPair:
        pair = MappingPair.create(
            feature_id=feature.id,
            experience_id=experience.id,
            feature_text=feature.description,
            experience_text=experience.description,
            annotation=annotation,
        )
        self.pairs.append(pair)
        self.touch()
        return pair


# ---------------------------------------------------------------------
# Cover Letter
# ---------------------------------------------------------------------

class Idea(BaseEntity):
    title: str
    description: Optional[str] = None
    mapping_pair_ids: List[str] = Field(default_factory=list)
    annotation: Optional[str] = None


class Paragraph(BaseEntity):
    order: int
    idea_ids: List[str] = Field(default_factory=list)
    purpose: Optional[str] = None  # e.g., "Opening", "Skills", "Closing"
    draft_text: Optional[str] = None


class CoverLetter(BaseEntity):
    job_id: str
    base_cv_id: str
    mapping_id: str
    derived_cv_id: Optional[str] = None
    paragraphs: List[Paragraph] = Field(default_factory=list)
    ideas: List[Idea] = Field(default_factory=list)

    def add_paragraph(self, idea_ids: List[str], purpose: str) -> Paragraph:
        para = Paragraph.create(order=len(self.paragraphs) + 1, idea_ids=idea_ids, purpose=purpose)
        self.paragraphs.append(para)
        self.touch()
        return para


# ---------------------------------------------------------------------
# Application & Interview Tracking
# ---------------------------------------------------------------------

class InterviewQuestion(BaseEntity):
    question: str
    answer: Optional[str] = None
    stage: Optional[str] = None


class InterviewStage(BaseEntity):
    name: str
    description: Optional[str] = None
    status: Literal["pending", "completed"] = "pending"
    questions: List[InterviewQuestion] = Field(default_factory=list)


class Interview(BaseEntity):
    application_id: str
    current_stage: Optional[str] = None
    stages: List[InterviewStage] = Field(default_factory=list)

    def add_stage(self, name: str, description: Optional[str] = None) -> InterviewStage:
        stage = InterviewStage.create(name=name, description=description)
        self.stages.append(stage)
        self.current_stage = name
        self.touch()
        return stage


class Application(BaseEntity):
    job_id: str
    base_cv_id: str
    mapping_id: Optional[str] = None
    derived_cv_id: Optional[str] = None
    cover_letter_id: Optional[str] = None
    interview_id: Optional[str] = None
    status: Literal["draft", "applied", "interview", "offer", "rejected"] = "draft"
    notes: Optional[str] = None


# You already have BaseEntity and gen_id in your core
# We'll just build on that structure

class WorkItem(BaseEntity):
    """Represents a discrete piece of work done during the job search process."""

    title: str
    type: Literal[
        "research",
        "cv_update",
        "application",
        "interview_prep",
        "coding_test",
        "learning",
        "networking",
        "reflection",
    ] = "research"

    related_application_id: Optional[str] = None
    related_interview_id: Optional[str] = None
    related_job_id: Optional[str] = None
    related_skill_ids: List[str] = Field(default_factory=list)
    related_goal_id: Optional[str] = None

    status: Literal["planned", "in_progress", "completed"] = "planned"
    effort_hours: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    reflection: Optional[str] = None  # User-written reflection or insight
    outcome: Optional[str] = None  # e.g. “Improved confidence”, “Learned X library”

    def mark_completed(self, reflection: Optional[str] = None):
        """Mark as completed and optionally add a reflection."""
        self.status = "completed"
        self.reflection = reflection or self.reflection
        self.touch()


class Goal(BaseEntity):
    """Represents a larger focus area or objective grouping related WorkItems."""

    title: str
    description: Optional[str] = None
    metric: Optional[str] = None  # e.g. "Finish 3 coding tests"
    progress: float = 0.0  # 0.0–1.0 completion
    work_item_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    status: Literal["active", "paused", "completed"] = "active"
    due_date: Optional[datetime] = None
    reflection: Optional[str] = None

    def add_work_item(self, work_item: WorkItem):
        """Link a WorkItem to this Goal."""
        if work_item.id not in self.work_item_ids:
            self.work_item_ids.append(work_item.id)
            work_item.related_goal_id = self.id
            self.touch()

    def update_progress(self, completed_work_items: int, total_work_items: int):
        """Recalculate progress manually or based on related WorkItems."""
        if total_work_items > 0:
            self.progress = round(completed_work_items / total_work_items, 2)
        else:
            self.progress = 0.0
        self.touch()

    def mark_completed(self, reflection: Optional[str] = None):
        """Mark the Goal as completed."""
        self.status = "completed"
        self.reflection = reflection or self.reflection
        self.progress = 1.0
        self.touch()


# ---------------------------------------------------------------------
# API Update Models (for PATCH operations)
# ---------------------------------------------------------------------

class CVUpdate(BaseModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None
    # Nested lists like experiences or skills require dedicated sub-routes/registry methods to manage properly, 
    # so we don't include List[T] here for partial updates.

class JobDescriptionUpdate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None

class ApplicationStatus(BaseModel):
    """A minimal model used specifically for the PUT /application/{id}/status route."""
    status: Literal["draft", "applied", "interview", "offer", "rejected"]

class ApplicationUpdate(BaseModel):
    status: Optional[Literal["draft", "applied", "interview", "offer", "rejected"]] = None
    notes: Optional[str] = None

class MappingUpdate(BaseModel):
    # This model is primarily a placeholder for consistency; core changes are via mapping pairs.
    pass

class WorkItemUpdate(BaseModel):
    title: Optional[str] = None
    type: Optional[Literal[
        "research",
        "cv_update",
        "application",
        "interview_prep",
        "coding_test",
        "learning",
        "networking",
        "reflection",
    ]] = None
    status: Optional[Literal["planned", "in_progress", "completed"]] = None
    effort_hours: Optional[float] = None
    tags: Optional[List[str]] = None
    reflection: Optional[str] = None
    outcome: Optional[str] = None
    # Relationship IDs can be handled via separate linking routes (e.g., add_work_to_goal)

class GoalUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    metric: Optional[str] = None
    status: Optional[Literal["active", "paused", "completed"]] = None
    due_date: Optional[datetime] = None
    reflection: Optional[str] = None

# --- NESTED ITEM UPDATE MODELS ---

class SkillUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[Literal["technical", "soft", "language", "other"]] = None
    level: Optional[str] = None
    importance: Optional[int] = Field(default=None, ge=1, le=5)
    description: Optional[str] = None

# Add near other update models
class ExperienceUpdate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    skill_ids: Optional[List[str]] = None # Allow updating skills
    achievement_ids: Optional[List[str]] = None # Allow updating achievements

class EducationUpdate(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    related_experience_id: Optional[str] = None
    related_education_id: Optional[str] = None

class HobbyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class AchievementUpdate(BaseModel):
    text: Optional[str] = None
    context: Optional[str] = None

class JobFeatureUpdate(BaseModel):
    type: Optional[Literal["requirement", "responsibility", "value", "nice_to_have", "qualification", "benefit", "other"]] = None
    description: Optional[str] = None

class MappingPairUpdate(BaseModel):
    annotation: Optional[str] = None

class IdeaUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    annotation: Optional[str] = None

class ParagraphUpdate(BaseModel):
    order: Optional[int] = None
    purpose: Optional[str] = None
    draft_text: Optional[str] = None

class InterviewStageUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[Literal["pending", "completed"]] = None

class InterviewQuestionUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    stage: Optional[str] = None


# ---------------------------------------------------------------------
# AI Prompt Generation Models
# ---------------------------------------------------------------------

class CVGenerationPrompt(BaseModel):
    """Structured data payload for a CV generation LLM service."""
    instruction: str = Field(description="System instruction to the LLM to act as a career assistant.")
    job_description: JobDescription
    base_cv: CV
    mapping_data: Mapping
    goal: str = Field(description="Generate a derived CV by prioritizing experience in the CV that is linked to job requirements in the mapping_data.")

class CoverLetterGenerationPrompt(BaseModel):
    """Structured data payload for a Cover Letter generation LLM service."""
    instruction: str = Field(description="System instruction to the LLM to act as a copywriter generating a professional cover letter.")
    job_description: JobDescription
    base_cv: CV
    mapping_data: Mapping
    cover_letter_ideas: List[Idea] = Field(default_factory=list)
    goal: str = Field(description="Draft a cover letter using the cover_letter_ideas, ensuring all points are relevant to the job_description features linked in mapping_data.")

class AIPromptResponse(BaseModel):
    """The unified response model for AI generation endpoints."""
    job_id: str
    cv_id: str
    prompt_type: Literal["CV", "CoverLetter"]
    structured_payload: Union[CVGenerationPrompt, CoverLetterGenerationPrompt]