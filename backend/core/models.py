# backend/core/models.py

from __future__ import annotations
from typing import List, Dict, Optional, Literal, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from dateutil import parser as date_parser
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


# backend/core/models.py

class UserOwnedEntity(BaseEntity):
    """
    Mixin for 'Root' entities that must belong to a specific user.
    Replaces BaseEntity for top-level tables.
    """
    user_id: str

# ---------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------

class Skill(BaseEntity):
    name: str
    category: str = "technical"
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


# --- UPDATED DATE NORMALIZER ---
def normalize_date_string(v: Optional[str], default_month: int = 1) -> Optional[str]:
    """
    Cleans date strings.
    - "Present", "Now" -> None
    - "2023" -> "2023-01-01" (if default_month=1) or "2023-09-01" (if default_month=9)
    - Fallback: returns original string if parsing fails.
    """
    if not v:
        return None
    
    s = str(v).strip().lower()
    
    # Handle "Present" keywords -> Null (Ongoing)
    if s in ["present", "now", "current", "null", "none", "ongoing"]:
        return None

    # print(f"Attempting to parse date string: '{v}' with default month {default_month}"  )
        
    try:
        # We use a default year of 2000, but the critical part is default_month and day=1.
        # This prevents the parser from using "Today's" month/day.
        default_date = datetime(2000, default_month, 1)
        
        dt = date_parser.parse(v, default=default_date)
        # print(f"Parsed date '{v}' as '{dt.date()}'" )
        return dt.strftime("%Y-%m-%d")
    except:
        # print(f"Warning: Failed to parse date '{v}'. Keeping original string.")
        return v

# ---------------------------------------------------------------------
# Experience, Education, Project, Hobby
# ---------------------------------------------------------------------

class Experience(BaseEntity, SkillLinkMixin):
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    achievement_ids: List[str] = Field(default_factory=list)

    # --- 2. Add Validators ---
    @validator('start_date', 'end_date', pre=True)
    def clean_dates(cls, v):
        return normalize_date_string(v, default_month=1)

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

    # --- 2. Add Validators ---
    @validator('start_date', 'end_date', pre=True)
    def clean_dates(cls, v):
        return normalize_date_string(v, default_month=9)

    def add_achievement(self, achievement: Achievement):
        if achievement.id not in self.achievement_ids:
            self.achievement_ids.append(achievement.id)
        self.touch()
        return achievement


class Project(BaseEntity, SkillLinkMixin):
    title: str
    description: str

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # DEPECREATED FIELDS: kept only for backward compatibility
    related_experience_id: Optional[str] = None
    related_education_id: Optional[str] = None

    # --- ADD THESE NEW FIELDS ---
    related_experience_ids: List[str] = Field(default_factory=list)
    related_education_ids: List[str] = Field(default_factory=list)
    related_hobby_ids: List[str] = Field(default_factory=list)  # <--- Added Hobbies support

    achievement_ids: List[str] = Field(default_factory=list)

     # --- 2. Add Validators ---
    @validator('start_date', 'end_date', pre=True)
    def clean_dates(cls, v):
        return normalize_date_string(v, default_month=1)


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

class CV(UserOwnedEntity):
    """Base CV containing all content."""
    name: str
    # --- NEW FIELDS ---
    first_name: Optional[str] = None # Actual first name for the document
    last_name: Optional[str] = None  # Actual last name for the document
    title: Optional[str] = None   # e.g., "Mr/Dr/Ms/Mrs/Miss"
    is_title_in_name: Optional[bool] = False  # If True, "title" is part of "name" and should be rendered as such
    # --- END NEW FIELDS ---
    summary: Optional[str] = None
    contact_info: Dict[str, str] = Field(default_factory=dict)
    experiences: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    hobbies: List[Hobby] = Field(default_factory=list)
    achievements: List[Achievement] = Field(default_factory=list)

    # --- NEW FIELDS FOR BACKGROUND TASKS ---
    is_importing: Optional[bool] = False
    import_task_id: Optional[str] = None
    # ---------------------------------------

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

# backend/core/models.py

# ... (keep all your existing imports and other classes) ...

class DerivedCV(CV):
    """Job-specific tailored CV derived from a base CV."""
    base_cv_id: str
    job_id: str
    mapping_id: str
    application_id: Optional[str] = None
    
    selected_experience_ids: List[str] = Field(default_factory=list)
    selected_skill_ids: List[str] = Field(default_factory=list)
    selected_project_ids: List[str] = Field(default_factory=list)
    selected_education_ids: List[str] = Field(default_factory=list)
    selected_hobby_ids: List[str] = Field(default_factory=list)
    
    is_locked: bool = False

    @classmethod
    def from_mapping(cls, base_cv: CV, job_id: str, mapping: Mapping) -> 'DerivedCV':
        """
        Creates a DEEP COPY of the Base CV with intelligent selection logic.
        Reflects hierarchical dependencies:
        1. Direct AI Matches.
        2. Child -> Parent (e.g. Active Project -> Activates related Experience).
        3. Parent -> Skills (e.g. Active Hobby -> Activates its Skills).
        4. Parent -> Achievements (Active Job -> Activates its bullet points).
        """
        
        # --- 1. DEEP COPY EVERYTHING ---
        # We must create independent copies so editing the Derived CV doesn't touch the Base CV.
        # However, IDs remain the same to preserve internal relationships.
        def clone_list(items):
            return [item.model_copy(deep=True) for item in items]

        cloned_experiences = clone_list(base_cv.experiences)
        cloned_education = clone_list(base_cv.education)
        cloned_projects = clone_list(base_cv.projects)
        cloned_hobbies = clone_list(base_cv.hobbies)
        cloned_skills = clone_list(base_cv.skills)
        cloned_achievements = clone_list(base_cv.achievements)

        # --- 2. BUILD LOOKUP MAPS ---
        exp_map = {e.id: e for e in cloned_experiences}
        edu_map = {e.id: e for e in cloned_education}
        proj_map = {p.id: p for p in cloned_projects}
        hobby_map = {h.id: h for h in cloned_hobbies}
        ach_map = {a.id: a for a in cloned_achievements}
        skill_map = {s.id: s for s in cloned_skills}

        # Reverse Map for Achievements: AchID -> List[ParentItems]
        # (Since Achievement doesn't store its parent ID, we must map it for upward propagation)
        ach_parents = {}
        def register_ach_parent(parent_item):
            for ach_id in parent_item.achievement_ids:
                if ach_id not in ach_parents: ach_parents[ach_id] = []
                ach_parents[ach_id].append(parent_item)

        for item in cloned_experiences: register_ach_parent(item)
        for item in cloned_education: register_ach_parent(item)
        for item in cloned_projects: register_ach_parent(item)
        for item in cloned_hobbies: register_ach_parent(item)

        # --- 3. SEED INITIAL SELECTION ---
        # Get all IDs the AI explicitly matched
        mapped_ids = {p.context_item_id for p in mapping.pairs if p.context_item_id}

        # Initialize Selection Sets
        sel_exp = {mid for mid in mapped_ids if mid in exp_map}
        sel_edu = {mid for mid in mapped_ids if mid in edu_map}
        sel_proj = {mid for mid in mapped_ids if mid in proj_map}
        sel_hobby = {mid for mid in mapped_ids if mid in hobby_map}
        sel_ach = {mid for mid in mapped_ids if mid in ach_map}
        sel_skill = {mid for mid in mapped_ids if mid in skill_map}

        # --- 4. PROPAGATION LOOP ---
        # We loop until no new items are added to ensure complex chains 
        # (e.g., Skill -> Achievement -> Project -> Experience) are fully resolved.
        changes = True
        while changes:
            changes = False
            
            # Snapshot current counts to detect changes
            current_counts = (len(sel_exp), len(sel_edu), len(sel_proj), len(sel_hobby), len(sel_ach), len(sel_skill))

            # --- A. UPWARD PROPAGATION (Child -> Parent) ---
            
            # 1. If Achievement is active -> Activate its Parent(s)
            for ach_id in list(sel_ach):
                parents = ach_parents.get(ach_id, [])
                for p in parents:
                    if isinstance(p, Experience) and p.id not in sel_exp:
                        sel_exp.add(p.id)
                    elif isinstance(p, Education) and p.id not in sel_edu:
                        sel_edu.add(p.id)
                    elif isinstance(p, Project) and p.id not in sel_proj:
                        sel_proj.add(p.id)
                    elif isinstance(p, Hobby) and p.id not in sel_hobby:
                        sel_hobby.add(p.id)

            # 2. If Project is active -> Activate its Context (Parents)
            for proj_id in list(sel_proj):
                proj = proj_map[proj_id]
                # Related Experiences
                for pid in proj.related_experience_ids:
                    if pid in exp_map and pid not in sel_exp:
                        sel_exp.add(pid)
                # Related Education
                for eid in proj.related_education_ids:
                    if eid in edu_map and eid not in sel_edu:
                        sel_edu.add(eid)
                # Related Hobbies
                for hid in proj.related_hobby_ids:
                    if hid in hobby_map and hid not in sel_hobby:
                        sel_hobby.add(hid)

            # --- B. DOWNWARD PROPAGATION (Parent -> Content) ---
            
            # Collect all currently active parent entities
            active_entities = []
            active_entities.extend([exp_map[i] for i in sel_exp])
            active_entities.extend([edu_map[i] for i in sel_edu])
            active_entities.extend([proj_map[i] for i in sel_proj])
            active_entities.extend([hobby_map[i] for i in sel_hobby])
            # Also include achievements because they contain skills
            active_entities.extend([ach_map[i] for i in sel_ach])

            for entity in active_entities:
                # 3. If Entity is active -> Activate its Skills
                if hasattr(entity, 'skill_ids'):
                    for sid in entity.skill_ids:
                        if sid in skill_map and sid not in sel_skill:
                            sel_skill.add(sid)

                # 4. If Entity is active -> Activate its Achievements
                # (Standard CV logic: If I include a Job, I generally want its bullet points)
                if hasattr(entity, 'achievement_ids'):
                    for aid in entity.achievement_ids:
                        if aid in ach_map and aid not in sel_ach:
                            sel_ach.add(aid)

            # Check if anything changed
            new_counts = (len(sel_exp), len(sel_edu), len(sel_proj), len(sel_hobby), len(sel_ach), len(sel_skill))
            if new_counts != current_counts:
                changes = True

        return cls.create(
            user_id=base_cv.user_id, # <--- Fix: Pass User ID
            name=f"Tailored for {job_id[:8]}",
            base_cv_id=base_cv.id,
            job_id=job_id,
            mapping_id=mapping.id,
            
            # Content (Full Clones)
            first_name=base_cv.first_name, 
            last_name=base_cv.last_name, 
            title=base_cv.title, 
            summary=base_cv.summary,
            contact_info=base_cv.contact_info.copy() if base_cv.contact_info else {},
            experiences=cloned_experiences,
            education=cloned_education,
            projects=cloned_projects,
            hobbies=cloned_hobbies,
            skills=cloned_skills,
            achievements=cloned_achievements,
            
            # Metadata (The Filter)
            selected_experience_ids=list(sel_exp),
            selected_education_ids=list(sel_edu),
            selected_project_ids=list(sel_proj),
            selected_hobby_ids=list(sel_hobby),
            selected_skill_ids=list(sel_skill),
        )   
# ---------------------------------------------------------------------
# Job description & matching
# ---------------------------------------------------------------------

class CVExportRequest(BaseModel):
    # Added "hobbies" to default list
    section_order: List[str] = ["education", "skills", "projects", "experience", "hobbies"]
    section_titles: Dict[str, str] = {
        "education": "Education",
        "skills": "Technical Skills",
        "projects": "Academic & Research Projects",
        "experience": "Experience",
        "hobbies": "Interests & Hobbies"
    }
    file_format: Literal["pdf", "docx", "tex", "zip"] = "pdf"


class CVImportRequest(BaseModel):
    text: str
    name: Optional[str] = "Imported CV"


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


class JobDescription(UserOwnedEntity):
    title: str
    company: str
    features: List[JobDescriptionFeature] = Field(default_factory=list)
    notes: Optional[str] = None

    # --- NEW FIELDS ---
    job_url: Optional[str] = None
    
    location: Optional[str] = None
    salary_range: Optional[str] = None
    # --- END NEW FIELDS ---

    # --- INTELLIGENCE CACHE (For Spotlight & Sorting) ---
    match_score: float = 0.0 
    
    # "Green", "Yellow", "Red" status based on the score
    match_grade: Literal["A", "B", "C", "D"] = "D" 
    
    # Badges for the UI card (e.g. "Missing Degree", "High Salary")
    cached_badges: List[str] = Field(default_factory=list)
    # ----------------------------------------------------


    application_end_date: Optional[str] = None # deprecated but identical to date_closing
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
    The all-in-one payload to create or fully update a job.
    Used by POST /job/upsert.
    """
    id: Optional[str] = None # If null, create new job. If set, update job.
    title: str
    company: str
    
    job_url: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    notes: Optional[str] = None

    # Date handling
    application_end_date: Optional[str] = None # Deprecated alias for date_closing
    date_closing: Optional[str] = None 
    date_posted: Optional[str] = None
    date_extracted: Optional[str] = None
    
    # Content
    description: Optional[str] = None           
    displayed_description: Optional[str] = None 
    
    # Features List
    # The backend will replace the old list with this new one for the given Job ID.
    features: List[JobFeatureInput] = Field(default_factory=list)

# 1. Atomic Lineage Step (The Breadcrumb)
class LineageItem(BaseModel):
    id: str
    type: str   # e.g., "project", "achievement"
    name: str   # e.g., "E-Commerce App"

# 2. A Single Segment Match (The Evidence Atom)
class MatchCandidate(BaseModel):
    segment_text: str          # The exact text matched
    segment_type: str          # e.g., "description", "title"
    score: float               # Confidence for this specific segment
    lineage: List[LineageItem] # Full path to this segment

# The Meta Container (Stores the "Why")
class MatchingMeta(BaseEntity):
    # The primary evidence that won the match
    best_match: MatchCandidate
    
    # Other segments in the same item that also matched (Supporting Evidence)
    # e.g. You matched the Title, but also matched a Bullet Point.
    supporting_matches: List[MatchCandidate] = Field(default_factory=list)

    # NEW: Keep track of what the user hated
    rejected_matches: List[MatchCandidate] = Field(default_factory=list)
    
    # Forensic note for quick display
    summary_note: str


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

    meta: Optional[MatchingMeta] = None

    # --- NEW FIELDS FOR SMART MAPPING ---
    # 'ai_proposed': Standard AI guess. Overwritable.
    # 'user_approved': User clicked "Yes". Locked against weak AI updates.
    # 'user_manual': User manually linked it. Strictly Locked.
    status: Literal["ai_proposed", "user_approved", "user_manual"] = "ai_proposed"
    
    # Store hashes of rejected content to prevent "Zombie Matches"
    # (Matches returning from the dead after re-inference)
    rejected_match_hashes: List[str] = Field(default_factory=list)
    # ------------------------------------



class Mapping(UserOwnedEntity):
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


class CoverLetter(UserOwnedEntity):
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


class Interview(UserOwnedEntity):
    application_id: str
    current_stage: Optional[str] = None
    stages: List[InterviewStage] = Field(default_factory=list)

    def add_stage(self, name: str, description: Optional[str] = None) -> InterviewStage:
        stage = InterviewStage.create(name=name, description=description)
        self.stages.append(stage)
        self.current_stage = name
        self.touch()
        return stage


class Application(UserOwnedEntity):
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

    # --- NEW CACHE FIELDS ---
    # Stores the score specific to THIS application's CV
    match_score: float = 0.0 
    match_grade: Literal["A", "B", "C", "D"] = "D"
    cached_badges: List[str] = Field(default_factory=list)


# You already have BaseEntity and gen_id in your core
# We'll just build on that structure

class WorkItem(UserOwnedEntity):
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


class Goal(UserOwnedEntity):
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
    category: str = "technical"

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

    start_date: Optional[str] = None
    end_date: Optional[str] = None

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
    """
    Model for updating a CV (Base or Derived).
    All fields are optional to allow partial updates (PATCH) 
    or full replacements (PUT).
    """
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None
    
    # Nested Lists - Replacing the list entirely is usually safer for reordering
    experiences: Optional[List[Experience]] = None
    education: Optional[List[Education]] = None
    skills: Optional[List[Skill]] = None
    projects: Optional[List[Project]] = None
    hobbies: Optional[List[Hobby]] = None
    achievements: Optional[List[Achievement]] = None

    # Derived CV specific fields
    selected_experience_ids: Optional[List[str]] = None
    selected_skill_ids: Optional[List[str]] = None
    selected_project_ids: Optional[List[str]] = None
    selected_education_ids: Optional[List[str]] = None
    selected_hobby_ids: Optional[List[str]] = None

    # Helper to filter out None values for patching
    def items_set(self):
        return self.dict(exclude_unset=True)

class JobDescriptionUpdate(BaseModel):
    """
    Validation schema for PATCH /job/{id}.
    Allows partial updates to any specific field without requiring the full object.
    """
    title: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None
    
    # --- New Fields Added to Support Frontend Editing ---
    job_url: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    
    date_closing: Optional[str] = None
    date_posted: Optional[str] = None
    date_extracted: Optional[str] = None # Usually backend-only, but allowed for manual overrides
    
    description: Optional[str] = None
    displayed_description: Optional[str] = None

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
    # --- ADD THIS IF MISSING ---
    derived_cv_id: Optional[str] = None 
    # ---------------------------
    
    supporting_document_ids: Optional[List[str]] = None
    match_score: Optional[float] = None
    match_grade: Optional[str] = None
    cached_badges: Optional[List[str]] = None
    is_locked: Optional[bool] = None

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



# backend/core/models.py (Append this to the end)

# ---------------------------------------------------------------------
# Forensic Analysis Models (The "RoleCase" Engine)
# ---------------------------------------------------------------------

class JobFitStats(BaseModel):
    overall_match_score: float
    coverage_pct: float
    average_quality: float
    
    # Counts
    total_reqs: int
    met_reqs: int
    critical_gaps_count: int
    
    # Navigation / Deep Dive Data
    # List of IDs for the "Warning" icon click-through
    missing_critical_ids: List[str] 
    
    # Authority Breakdown (e.g. {"Professional": 5, "Academic": 2})
    evidence_sources: Dict[str, int]
    evidence_ids_by_source: Dict[str, List[str]] 


class ForensicAlternative(BaseModel):
    """A rich summary of a supporting match."""
    id: str               # Unique ref for actions (hash or uuid)
    match_text: str       
    score: float          
    source_type: str      
    source_name: str      
    # --- NEW FIELDS ---
    lineage: List[LineageItem] = Field(default_factory=list) 
    cv_item_id: Optional[str] = None


class ForensicItem(BaseModel):
    requirement_id: str
    requirement_text: str
    requirement_type: str 
    importance: str
    status: Literal["verified", "pending", "missing"]
    
    # --- EVIDENCE NAVIGATION ---
    best_match_id: Optional[str] = None
    cv_item_id: Optional[str] = None
    cv_item_type: Optional[str] = None
    
    # --- INTERACTIVITY SUPPORT (NEW) ---
    # This carries the IDs (e.g. {'id': 'exp_123', 'type': 'experience'})
    # so the UI can render: [Experience] -> [Project] -> [Skill] as clickable chips.
    lineage: List[LineageItem] = Field(default_factory=list) 
    
    # --- DISPLAY DATA ---
    best_match_text: Optional[str] = None
    best_match_excerpt: Optional[str] = None
    best_match_confidence: float = 0.0
    
    # Renamed from 'lineage_text' to be more accurate
    match_summary: Optional[str] = None 
    
    authority_bucket: str = "Missing"

    alternatives: List[ForensicAlternative] = Field(default_factory=list)

class ForensicAnalysis(BaseModel):
    """The complete RoleCase report."""
    # Optional: The calculator doesn't need to know this, but the UI might want it.
    application_id: Optional[str] = None 
    stats: JobFitStats
    # Grouped for UI columns (Key = Importance Label: "Critical", "High", etc.)
    groups: Dict[str, List[ForensicItem]]
    suggested_grade: str = "D"
    suggested_badges: List[str] = []


# ---------------------------------------------------------------------
# Login & User Management Models
# --------------------------------------------------------------------- 
class User(BaseEntity):
    email: str
    oauth_provider: str = "google" 
    id: str
    provider_id: str
    full_name: str

    primary_cv_id: Optional[str] = None
    avatar_url: Optional[str] = None
    
    # Freemium Logic
    tier: Literal["free", "pro", "admin"] = "free"
    is_active: bool = True
    
    # Quotas (e.g. resets daily)
    quota_generations_used: int = 0
    quota_limit: int = 3 # Free tier limit
    last_quota_reset: datetime = Field(default_factory=datetime.utcnow)

