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
        # --- THIS IS THE FIX ---
        # If 'id' is passed in as None, remove it so the
        # default_factory can run. If a real ID is passed, use it.
        if 'id' in kwargs and kwargs['id'] is None:
            del kwargs['id']
        elif 'id' not in kwargs:
             kwargs['id'] = gen_id(cls.__name__.lower())
        
        obj = cls(**kwargs)
        # Ensure the ID prefix is correct if a new ID was generated
        if not kwargs.get('id'):
             obj.id = gen_id(cls.__name__.lower())
        
        return obj
        # --- END OF FIX ---


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

    # DEPECREATED FIELDS: kept only for backward compatibility
    related_experience_id: Optional[str] = None
    related_education_id: Optional[str] = None

    # --- ADD THESE NEW FIELDS ---
    related_experience_ids: List[str] = Field(default_factory=list)
    related_education_ids: List[str] = Field(default_factory=list)
    related_hobby_ids: List[str] = Field(default_factory=list)  # <--- Added Hobbies support

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
    # --- NEW FIELDS ---
    first_name: Optional[str] = None # Actual first name for the document
    last_name: Optional[str] = None  # Actual last name for the document
    title: Optional[str] = None   # e.g., "Mr/Dr/Ms/Mrs/Miss"
    # --- END NEW FIELDS ---
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

    # *** MODIFICATION ***
    # This now accepts **kwargs which will be used to pass skill_ids and achievement_ids
    def add_experience(self, title: str, company: str, **kwargs) -> Experience:
        exp = Experience.create(title=title, company=company, **kwargs)
        self.experiences.append(exp)
        self.touch()
        return exp
    # *** END MODIFICATION ***

    # --- *** MODIFICATION: Update add_project to accept **kwargs *** ---
    def add_project(self, title: str, description: str, **kwargs) -> Project:
        proj = Project.create(title=title, description=description, **kwargs)
        self.projects.append(proj)
        self.touch()
        return proj
    # --- *** END MODIFICATION *** ---
    
    # --- *** NEWLY ADDED METHOD *** ---
    def add_education(self, institution: str, degree: str, **kwargs) -> Education:
        edu = Education.create(institution=institution, degree=degree, **kwargs)
        self.education.append(edu)
        self.touch()
        return edu

    def add_hobby(self, name: str, description: Optional[str] = None, **kwargs) -> Hobby:
        hobby = Hobby.create(name=name, description=description, **kwargs) # <-- This line is changed
        self.hobbies.append(hobby)
        self.touch()
        return hobby

    def add_achievement(self, text: str, context: Optional[str] = None, **kwargs) -> Achievement:
        # **kwargs will be used to pass skill_ids
        ach = Achievement.create(text=text, context=context, **kwargs)
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
    is_locked: bool = False  # Snapshot state

    @classmethod
    def from_mapping(cls, base_cv: CV, job_id: str, mapping: Mapping) -> DerivedCV:
        exp_ids = {p.experience_id for p in mapping.pairs if p.experience_id}
        skill_ids = set()
        for exp in base_cv.experiences:
            if exp.id in exp_ids:
                skill_ids.update(exp.skill_ids)

        return cls.create(
            name=f"{base_cv.name} — tailored for {job_id}",
            first_name=base_cv.first_name, # Pass explicit name
            last_name=base_cv.last_name, # Pass explicit name
            title=base_cv.title, # Pass explicit title
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
        "responsibility",
        "hard_skill",
        "soft_skill",
        "experience",
        "qualification",
        "requirement",
        "nice_to_have",
        "employer_mission",
        "employer_culture",
        "role_value",
        "benefit",
        "other",
    ] = "requirement"
    description: str


class JobDescription(BaseEntity):
    title: str
    company: str
    features: List[JobDescriptionFeature] = Field(default_factory=list)
    notes: Optional[str] = None

    # --- NEW FIELDS ---
    job_url: Optional[str] = None
    application_end_date: Optional[str] = None # Using str for YYYY-MM-DD from form
    location: Optional[str] = None
    salary_range: Optional[str] = None
    notes: Optional[str] = None
    # --- END NEW FIELDS ---

    date_closing: Optional[str] = None 
    date_posted: Optional[str] = None      # <--- ADDED
    date_extracted: Optional[str] = None   # <--- ADDED
    
    # --- CONTENT ---
    description: Optional[str] = None           # <--- ADDED: Raw text
    displayed_description: Optional[str] = None # <--- ADDED: HTML/Formatted

    def add_feature(self, description: str, type: str = "requirement") -> JobDescriptionFeature:
        f = JobDescriptionFeature.create(description=description, type=type)
        self.features.append(f)
        self.touch()
        return f

# --- NEW UPSERT PAYLOAD ---
class JobFeatureInput(BaseModel):
    """Payload for a single feature. ID is optional (for new features)."""
    id: Optional[str] = None # Will be null for new features
    type: Literal[
        "responsibility",
        "hard_skill",
        "soft_skill",
        "experience",
        "qualification",
        "requirement",
        "nice_to_have",
        "employer_mission",
        "employer_culture",
        "role_value",
        "benefit",
        "other",
    ] = "requirement"
    description: str

class JobUpsertPayload(BaseModel):
    """
    The all-in-one payload to create or update a job.
    The backend will handle creating new IDs for features.
    """
    id: Optional[str] = None # If null, create new job. If set, update job.
    title: str
    company: str
    job_url: Optional[str] = None
    application_end_date: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    notes: Optional[str] = None

    date_closing: Optional[str] = None 
    date_posted: Optional[str] = None      # <--- ADDED
    date_extracted: Optional[str] = None   # <--- ADDED
    
    # --- CONTENT ---
    description: Optional[str] = None           # <--- ADDED: Raw text
    displayed_description: Optional[str] = None # <--- ADDED: HTML/Formatted
    
    # The frontend will send the *full* list of features.
    # The backend will replace the old list with this new one.
    features: List[JobFeatureInput] = Field(default_factory=list)

class MappingPair(BaseEntity):
    feature_id: str
    # --- START CHANGES ---
    # experience_id: Optional[str] = None # <-- REMOVE THIS
    context_item_id: Optional[str] = None # <-- ADD THIS (e.g., 'exp_123', 'proj_456')
    context_item_type: Optional[str] = None # <-- ADD THIS (e.g., 'experiences', 'projects')
    # --- END CHANGES ---
    feature_text: Optional[str] = None
    # --- START CHANGES ---
    # experience_text: Optional[str] = None # <-- REMOVE THIS
    context_item_text: Optional[str] = None # <-- ADD THIS
    # --- END CHANGES ---

    # --- ADD THIS LINE ---
    strength: Optional[float] = None 
    # ---------------------
    
    annotation: Optional[str] = None


class Mapping(BaseEntity):
    job_id: str
    base_cv_id: str
    pairs: List[MappingPair] = Field(default_factory=list)
    is_locked: bool = False # <--- NEW

    # def add_pair(
    #     self,
    #     feature: JobDescriptionFeature,
    #     experience: Experience,
    #     annotation: Optional[str] = None,
    # ) -> MappingPair:
    #     pair = MappingPair.create(
    #         feature_id=feature.id,
    #         experience_id=experience.id,
    #         feature_text=feature.description,
    #         experience_text=experience.description,
    #         annotation=annotation,
    #     )
    #     self.pairs.append(pair)
    #     self.touch()
    #     return pair


# ---------------------------------------------------------------------
# Cover Letter
# ---------------------------------------------------------------------

class Idea(BaseEntity):
    title: str
    description: Optional[str] = None
    mapping_pair_ids: List[str] = Field(default_factory=list)
    annotation: Optional[str] = None
    # --- NEW FIELD ---
    # Stores IDs of raw CV items (e.g., 'exp_123', 'skill_456') linked to this argument
    related_entity_ids: List[str] = Field(default_factory=list) 
    # -----------------
    owner: Literal["user", "autofill"] = "user"
    classification: Literal["professional", "personal", "company", "unclassified"] = "unclassified"



class Paragraph(BaseEntity):
    order: int
    idea_ids: List[str] = Field(default_factory=list)
    purpose: Optional[str] = None  # e.g., "Opening", "Skills", "Closing"
    draft_text: Optional[str] = None
    owner: Literal["user", "autofill"] = "user"


class CoverLetter(BaseEntity):
    name: str = "Cover Letter"  # <--- NEW: User-defined name (e.g. "Selection Criteria")
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
    # CHANGED: Replaces singular cover_letter_id
    supporting_document_ids: List[str] = Field(default_factory=list)
    cover_letter_id: Optional[str] = None
    interview_id: Optional[str] = None
    status: Literal["draft", "applied", "interview", "offer", "rejected"] = "draft"
    notes: Optional[str] = None
    # NEW: Snapshot metadata
    is_locked: bool = False 
    applied_at: Optional[datetime] = None


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
# --- *** NEW: Complex Payload Models for Experience ***
# ---------------------------------------------------------------------

class PendingSkillInput(BaseModel):
    """Payload for a skill that needs to be created."""
    name: str
    category: Literal["technical", "soft", "language", "other"] = "technical"

class PendingAchievementInput(BaseModel):
    """Payload for an achievement that needs to be created."""
    text: str
    context: Optional[str] = "Global"
    original_id: Optional[str] = None # Used when modifying a master achievement
    existing_skill_ids: List[str] = Field(default_factory=list)
    new_skills: List[PendingSkillInput] = Field(default_factory=list)

class ExperienceComplexPayload(BaseModel):
    """
    The all-in-one payload from the frontend to create or update an experience
    and all its new/modified dependencies in one API call.
    """
    # Core Experience fields
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    
    # Direct Skills
    existing_skill_ids: List[str] = Field(default_factory=list)
    new_skills: List[PendingSkillInput] = Field(default_factory=list)
    
    # Achievements
    existing_achievement_ids: List[str] = Field(default_factory=list)
    new_achievements: List[PendingAchievementInput] = Field(default_factory=list)

# --- *** NEW: Complex Payload Model for Education *** ---

class EducationComplexPayload(BaseModel):
    """
    The all-in-one payload from the frontend to create or update an education
    and all its new/modified dependencies in one API call.
    """
    # Core Education fields
    institution: str
    degree: str
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Direct Skills
    existing_skill_ids: List[str] = Field(default_factory=list)
    new_skills: List[PendingSkillInput] = Field(default_factory=list)
    
    # Achievements
    existing_achievement_ids: List[str] = Field(default_factory=list)
    new_achievements: List[PendingAchievementInput] = Field(default_factory=list)

# --- *** NEW: Complex Payload Model for Hobby *** ---

class HobbyComplexPayload(BaseModel):
    """
    The all-in-one payload from the frontend to create or update a hobby
    and all its new/modified dependencies in one API call.
    """
    # Core Hobby fields
    name: str
    description: Optional[str] = None
    
    # Direct Skills
    existing_skill_ids: List[str] = Field(default_factory=list)
    new_skills: List[PendingSkillInput] = Field(default_factory=list)
    
    # Achievements
    existing_achievement_ids: List[str] = Field(default_factory=list)
    new_achievements: List[PendingAchievementInput] = Field(default_factory=list)

# --- *** NEW: Complex Payload Model for Project *** ---

class ProjectComplexPayload(BaseModel):
    """
    The all-in-one payload from the frontend to create or update a project
    and all its new/modified dependencies in one API call.
    """
    # Core Project fields
    title: str
    description: Optional[str] = None

    # DEPECREATED FIELDS: kept only for backward compatibility
    related_experience_id: Optional[str] = None
    related_education_id: Optional[str] = None

    # Add plurals (Optional for now so old frontend calls don't fail validation)
    related_experience_ids: List[str] = Field(default_factory=list)
    related_education_ids: List[str] = Field(default_factory=list)
    related_hobby_ids: List[str] = Field(default_factory=list)
    
    # Direct Skills
    existing_skill_ids: List[str] = Field(default_factory=list)
    new_skills: List[PendingSkillInput] = Field(default_factory=list)
    
    
    # Achievements
    existing_achievement_ids: List[str] = Field(default_factory=list)
    new_achievements: List[PendingAchievementInput] = Field(default_factory=list)



# --- *** END NEW MODEL *** ---
# --- *** END NEW MODEL *** ---

# ---------------------------------------------------------------------
# API Update Models (for PATCH operations)
# ---------------------------------------------------------------------

class CVUpdate(BaseModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    # --- NEW FIELDS ---
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    # --- END NEW FIELDS ---
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
    supporting_document_ids: List[str] = Field(default_factory=list)
    is_locked: bool = False 
    applied_at: Optional[datetime] = None
    cover_letter_id: Optional[str] = None # <-- ADD THIS LINE

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
    mapping_pair_ids: Optional[List[str]] = None
    # --- NEW FIELD ---
    related_entity_ids: Optional[List[str]] = None
    # -----------------
    annotation: Optional[str] = None
    # --- ADDED TO SUPPORT THE "OWNERSHIP & PROMOTION" RULE ---
    owner: Optional[Literal["user", "autofill"]] = None
    classification: Optional[Literal["professional", "personal", "company", "unclassified"]] = None


class ParagraphUpdate(BaseModel):
    order: Optional[int] = None
    idea_ids: Optional[List[str]] = None
    purpose: Optional[str] = None
    draft_text: Optional[str] = None
    # --- ADDED TO SUPPORT THE "OWNERSHIP & PROMOTION" RULE ---
    owner: Optional[Literal["user", "autofill"]] = None

class InterviewStageUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[Literal["pending", "completed"]] = None

class InterviewQuestionUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    stage: Optional[str] = None

class AppSuiteData(BaseModel):
    """
    A combined response model for fetching all data
    needed for the Application Suite view.
    """
    jobs: List[JobDescription]
    applications: List[Application]

class CoverLetterUpdate(BaseModel):
    name: Optional[str] = None # Allow renaming

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


# ---------------------------------------------------------------------
# PROMPT PAYLOAD MODELS (The Context Assembler Schema)
# ---------------------------------------------------------------------

class PromptReferenceItem(BaseModel):
    """
    A single normalized unit of data (Job Feature or CV Item).
    The AI uses this 'Bank' to look up details by ID.
    """
    id: str
    type: Literal["experience", "education", "project", "skill", "achievement", "hobby", "job_feature", "value", "unknown", "cv_item"]
    name: str
    detail: str 
    metadata: Optional[Dict[str, Any]] = None 

class PromptSegment(BaseModel):
    """A parsed chunk of the user's writing."""
    type: Literal["text", "context_injection", "structural_instruction"]
    content: str

class PromptEvidenceLink(BaseModel):
    """A connection the user has explicitly made (The 'Strong' Evidence)."""
    requirement_ref: Optional[str] = None # ID pointing to reference_bank
    evidence_ref: Optional[str] = None    # ID pointing to reference_bank
    annotation: Optional[str] = None      # Reasoning

class PromptArgument(BaseModel):
    """A strategic point within a paragraph."""
    topic: str
    user_strategy_notes: Optional[str] = None
    evidence_links: List[PromptEvidenceLink] = Field(default_factory=list)

class PromptParagraph(BaseModel):
    order: int
    purpose: str
    user_draft_segments: List[PromptSegment] = Field(default_factory=list)
    key_arguments: List[PromptArgument] = Field(default_factory=list) 

class CoverLetterPromptPayload(BaseModel):
    """
    The Master Payload.
    Contains EVERYTHING the AI needs to write a perfect letter.
    """
    # 1. The Stage (Job & Profile)
    job_context: Dict[str, Any]
    candidate_profile_summary: Dict[str, Any]
    
    # 2. The Database (All distinct entities, un-duplicated)
    reference_bank: Dict[str, PromptReferenceItem] 
    
    # 3. The Plan (User's explicit structure & notes)
    outline: List[PromptParagraph]
    
    # 4. The "Strong" Data Graph (Explicit Mappings)
    available_mappings: List[PromptEvidenceLink] = Field(default_factory=list)
    
    # 5. The "Unused" Potential (IDs that exist in bank but aren't mapped)
    # This tells the AI: "Here is extra stuff you can use if you need more evidence"
    unmapped_job_requirements: List[str] = Field(default_factory=list)
    unused_cv_items: List[str] = Field(default_factory=list)
    
    global_instructions: List[str]