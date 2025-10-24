from __future__ import annotations
from typing import List, Dict, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


# ---------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------

def gen_id(prefix: str) -> str:
    """Generate consistent unique IDs."""
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
# Core CV entities
# ---------------------------------------------------------------------

class Skill(BaseEntity):
    name: str
    category: Literal["technical", "soft", "language", "other"] = "technical"
    level: Optional[str] = None


class SkillLinkMixin(BaseModel):
    skill_ids: List[str] = Field(default_factory=list)

    def attach_skill(self, skill: Skill):
        if skill.id not in self.skill_ids:
            self.skill_ids.append(skill.id)


class Achievement(BaseEntity, SkillLinkMixin):
    text: str
    context: Optional[str] = None


class Experience(BaseEntity, SkillLinkMixin):
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    achievements: List[Achievement] = Field(default_factory=list)

    def add_achievement(self, text: str, context: Optional[str] = None):
        ach = Achievement.create(text=text, context=context)
        self.achievements.append(ach)
        return ach


class Education(BaseEntity, SkillLinkMixin):
    institution: str
    degree: str
    field: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    achievements: List[Achievement] = Field(default_factory=list)

    def add_achievement(self, text: str, context: Optional[str] = None):
        ach = Achievement.create(text=text, context=context)
        self.achievements.append(ach)
        return ach


class Project(BaseEntity, SkillLinkMixin):
    title: str
    description: str
    related_experience_id: Optional[str] = None
    related_education_id: Optional[str] = None
    achievements: List[Achievement] = Field(default_factory=list)

    def add_achievement(self, text: str, context: Optional[str] = None):
        ach = Achievement.create(text=text, context=context)
        self.achievements.append(ach)
        return ach


class Hobby(BaseEntity, SkillLinkMixin):
    name: str
    description: Optional[str] = None


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

    def add_experience(self, title: str, company: str, **kwargs) -> Experience:
        exp = Experience.create(title=title, company=company, **kwargs)
        self.experiences.append(exp)
        self.touch()
        return exp

    def add_skill(self, name: str, category: str = "technical", **kwargs) -> Skill:
        skill = Skill.create(name=name, category=category, **kwargs)
        self.skills.append(skill)
        self.touch()
        return skill

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
    def from_mapping(cls, base_cv: CV, job_id: str, mapping: Mapping) -> "DerivedCV":
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
    type: Literal["requirement", "responsibility", "value"] = "requirement"
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
    purpose: Optional[str] = None  # e.g. "Opening", "Skills", "Closing"
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
