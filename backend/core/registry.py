# backend/core/registry.py

from typing import Optional, List, Dict, Any
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
        # Note: self.db.insert maps to TinyDBManager.insert which performs upsert
        self.db.insert(table, obj)
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
        # Note: self.db.insert maps to TinyDBManager.insert which performs upsert
        self.db.insert(table, obj)

    def _delete(self, table: str, obj_id: str):
        """Delete an entity by ID."""
        if not self._get(table, BaseEntity, obj_id):
            raise ValueError(f"{table} with ID {obj_id} not found")
        self.db.remove(table, obj_id)
        return {"status": "success", "id": obj_id, "message": f"{table} deleted"}

    def _update_entity(self, table: str, cls: BaseModel, obj_id: str, update_data: dict):
        """Generic function to update an entity using data from an Update model."""
        obj = self._get(table, cls, obj_id)
        if not obj:
            raise ValueError(f"{cls.__name__} with ID {obj_id} not found")

        # Filter out None values from the update request data
        to_update = {k: v for k, v in update_data.items() if v is not None}

        # Apply updates to the model instance and update timestamp
        updated_data = obj.model_copy(update=to_update)
        updated_data.touch()
        self._update(table, updated_data)
        return updated_data


    # ---- Jobs ----
    def create_job(self, title: str, company: str, notes: Optional[str] = None):
        job = JobDescription.create(title=title, company=company, notes=notes)
        return self._insert("jobs", job)

    def update_job(self, job_id: str, update_data: JobDescriptionUpdate):
        return self._update_entity("jobs", JobDescription, job_id, update_data.model_dump())

    def delete_job(self, job_id: str):
        return self._delete("jobs", job_id)

    def add_job_feature(self, job_id: str, description: str, type: str = "requirement"):
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")
        feature = job.add_feature(description, type)
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

    def update_cv(self, cv_id: str, update_data: CVUpdate):
        return self._update_entity("cvs", CV, cv_id, update_data.model_dump())

    def delete_cv(self, cv_id: str):
        return self._delete("cvs", cv_id)

    def get_cv(self, cv_id: str):
        return self._get("cvs", CV, cv_id)

    def all_cvs(self):
        return self._all("cvs", CV)
    
    # --- NESTED ADD METHODS ---

    def add_cv_experience(self, cv_id: str, title: str, company: str, skill_ids: Optional[List[str]] = None, **kwargs) -> Experience:
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        # Pass all kwargs (like start_date, description) to the creator
        exp = cv.add_experience(title=title, company=company, **kwargs)
        
        # *** NEW: Set skill_ids if they were provided ***
        if skill_ids:
            exp.skill_ids = skill_ids
        
        self._update("cvs", cv)
        return exp

    def add_cv_education(self, cv_id: str, institution: str, degree: str, field: str, skill_ids: Optional[List[str]] = None, **kwargs) -> Education:
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        # Pass all kwargs (like start_date, end_date) to the creator
        edu = Education.create(institution=institution, degree=degree, field=field, **kwargs)
        
        # *** NEW: Set skill_ids if they were provided ***
        if skill_ids:
            edu.skill_ids = skill_ids
            
        cv.education.append(edu)
        cv.touch()
        self._update("cvs", cv)
        return edu

    def add_cv_skill(self, cv_id: str, name: str, category: str = "technical", **kwargs) -> Skill:
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        skill = cv.add_skill(name=name, category=category, **kwargs)
        self._update("cvs", cv)
        return skill

    def add_cv_project(self, cv_id: str, title: str, description: str, skill_ids: Optional[List[str]] = None, **kwargs) -> Project:
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        # Pass all kwargs (like related_experience_id) to the creator
        proj = cv.add_project(title=title, description=description, **kwargs)
        
        # *** NEW: Set skill_ids if they were provided ***
        if skill_ids:
            proj.skill_ids = skill_ids
            
        self._update("cvs", cv)
        return proj

    def add_cv_hobby(self, cv_id: str, name: str, description: Optional[str] = None, skill_ids: Optional[List[str]] = None) -> Hobby:
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        hobby = cv.add_hobby(name=name, description=description)
        
        # *** NEW: Set skill_ids if they were provided ***
        if skill_ids:
            hobby.skill_ids = skill_ids
            
        self._update("cvs", cv)
        return hobby


    def add_cv_achievement(self, cv_id: str, text: str, context: Optional[str] = None) -> Achievement:
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        ach = cv.add_achievement(text=text, context=context)
        self._update("cvs", cv)
        return ach

    # UPDATE methods
    def update_cv_experience(self, cv_id: str, exp_id: str, update_data: ExperienceUpdate):
        return self._update_nested_item(cv_id, 'experiences', exp_id, update_data.model_dump(exclude_none=True))
    
    def update_cv_education(self, cv_id: str, edu_id: str, update_data: EducationUpdate):
        return self._update_nested_item(cv_id, 'education', edu_id, update_data.model_dump(exclude_none=True))

    def update_cv_skill(self, cv_id: str, skill_id: str, update_data: SkillUpdate):
        return self._update_nested_item(cv_id, 'skills', skill_id, update_data.model_dump(exclude_none=True))

    def update_cv_achievement(self, cv_id: str, ach_id: str, update_data: AchievementUpdate):
        return self._update_nested_item(cv_id, 'achievements', ach_id, update_data.model_dump(exclude_none=True))

    def update_cv_project(self, cv_id: str, proj_id: str, update_data: ProjectUpdate):
        return self._update_nested_item(cv_id, 'projects', proj_id, update_data.model_dump(exclude_none=True))

    def update_cv_hobby(self, cv_id: str, hobby_id: str, update_data: HobbyUpdate):
        return self._update_nested_item(cv_id, 'hobbies', hobby_id, update_data.model_dump(exclude_none=True))

    # DELETE methods
    def delete_cv_experience(self, cv_id: str, exp_id: str):
        return self._delete_nested_item(cv_id, 'experiences', exp_id)
    
    def delete_cv_education(self, cv_id: str, edu_id: str):
        return self._delete_nested_item(cv_id, 'education', edu_id)

    def delete_cv_skill(self, cv_id: str, skill_id: str):
        # NOTE: Does not unlink from experiences/projects/achievements, which is fine for simplicity but a cleanup task for a full relational DB.
        return self._delete_nested_item(cv_id, 'skills', skill_id)

    def delete_cv_achievement(self, cv_id: str, ach_id: str):
        # NOTE: Does not unlink from experiences/projects/education, cleanup is expected on the frontend/future logic layer.
        return self._delete_nested_item(cv_id, 'achievements', ach_id)
    
    def delete_cv_project(self, cv_id: str, proj_id: str):
        return self._delete_nested_item(cv_id, 'projects', proj_id)
    
    def delete_cv_hobby(self, cv_id: str, hobby_id: str):
        return self._delete_nested_item(cv_id, 'hobbies', hobby_id)


    # --- NESTED LINKING METHODS ---

    def _get_nested_entity(self, cv: CV, entity_list_name: str, entity_id: str) -> Union[Experience, Education, Project, Hobby, Achievement]:
        """Helper to find a specific nested entity by ID within the CV."""
        if entity_list_name == 'skills':
             entity = next((e for e in cv.skills if e.id == entity_id), None)
        elif entity_list_name == 'experiences':
             entity = next((e for e in cv.experiences if e.id == entity_id), None)
        elif entity_list_name == 'education':
             entity = next((e for e in cv.education if e.id == entity_id), None)
        elif entity_list_name == 'projects':
             entity = next((e for e in cv.projects if e.id == entity_id), None)
        elif entity_list_name == 'hobbies':
             entity = next((e for e in cv.hobbies if e.id == entity_id), None)
        elif entity_list_name == 'achievements':
             entity = next((e for e in cv.achievements if e.id == entity_id), None)
        else:
             raise ValueError(f"Unknown entity type: {entity_list_name}")
        
        if not entity:
            raise ValueError(f"{entity_list_name.capitalize()} with ID {entity_id} not found in CV.")
        return entity

    def link_skill_to_entity(self, cv_id: str, entity_id: str, skill_id: str, entity_list_name: str):
        """Adds a skill ID to a specific nested entity's skill_ids list (Experience, Project, etc.)."""
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # 1. Find the target entity (Experience, Project, etc.)
        entity = self._get_nested_entity(cv, entity_list_name, entity_id)

        # 2. Check if the skill exists in the master CV list
        skill = self._get_nested_entity(cv, 'skills', skill_id)
        
        # 3. Perform the link (only works if entity has the SkillLinkMixin)
        if skill_id not in entity.skill_ids:
            entity.skill_ids.append(skill_id)
            cv.touch()
            self._update("cvs", cv)
        
        return entity

    def unlink_skill_from_entity(self, cv_id: str, entity_id: str, skill_id: str, entity_list_name: str):
        """Removes a skill ID from a specific nested entity's skill_ids list."""
        cv = self.get_cv(cv_id)
        if not cv: raise ValueError("CV not found")
        
        entity = self._get_nested_entity(cv, entity_list_name, entity_id)
        
        if skill_id in entity.skill_ids:
            entity.skill_ids.remove(skill_id)
            cv.touch()
            self._update("cvs", cv)
            return {"status": "success", "message": f"Skill {skill_id} unlinked from {entity_list_name[:-1]} {entity_id}."}
        
        raise ValueError(f"Skill {skill_id} was not linked to {entity_list_name[:-1]} {entity_id}.")
    
    def link_achievement_to_context(self, cv_id: str, context_id: str, ach_id: str, context_list_name: str):
        """Adds an achievement ID to a specific context's achievement_ids list (Experience, Project, etc.)."""
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # 1. Find the target context entity (Experience, Education, etc.)
        context = self._get_nested_entity(cv, context_list_name, context_id)

        # 2. Check if the achievement exists in the master CV list
        achievement = self._get_nested_entity(cv, 'achievements', ach_id)
        
        # 3. Perform the link
        if ach_id not in context.achievement_ids:
            context.achievement_ids.append(ach_id)
            cv.touch()
            self._update("cvs", cv)
        
        return context

    def unlink_achievement_from_context(self, cv_id: str, context_id: str, ach_id: str, context_list_name: str):
        """Removes an achievement ID from a specific context's achievement_ids list."""
        cv = self.get_cv(cv_id)
        if not cv: raise ValueError("CV not found")
        
        context = self._get_nested_entity(cv, context_list_name, context_id)
        
        if ach_id in context.achievement_ids:
            context.achievement_ids.remove(ach_id)
            cv.touch()
            self._update("cvs", cv)
            return {"status": "success", "message": f"Achievement {ach_id} unlinked from {context_list_name[:-1]} {context_id}."}
        
        raise ValueError(f"Achievement {ach_id} was not linked to {context_list_name[:-1]} {context_id}.")

    # ---- Mappings ----
    def create_mapping(self, job_id: str, base_cv_id: str):
        mapping = Mapping.create(job_id=job_id, base_cv_id=base_cv_id)
        return self._insert("mappings", mapping)

    def update_mapping(self, mapping_id: str, update_data: MappingUpdate):
        return self._update_entity("mappings", Mapping, mapping_id, update_data.model_dump())

    def delete_mapping(self, mapping_id: str):
        return self._delete("mappings", mapping_id)

    def add_mapping_pair(self, mapping_id: str, feature: JobDescriptionFeature, experience: Experience, annotation: Optional[str] = None):
        mapping = self.get_mapping(mapping_id)
        if not mapping:
            raise ValueError("Mapping not found")
        
        pair = mapping.add_pair(feature, experience, annotation)
        self._update("mappings", mapping)
        return pair

    def get_mapping(self, mapping_id: str):
        return self._get("mappings", Mapping, mapping_id)

    def all_mappings(self):
        return self._all("mappings", Mapping)
    

    # --- MAPPING PAIR NESTED CRUD ---
    def update_mapping_pair(self, mapping_id: str, pair_id: str, update_data: MappingPairUpdate):
        mapping = self.get_mapping(mapping_id)
        if not mapping: raise ValueError("Mapping not found")

        pair = next((p for p in mapping.pairs if p.id == pair_id), None)
        if not pair: raise ValueError("Mapping Pair not found")

        to_update = update_data.model_dump(exclude_none=True)
        for key, value in to_update.items():
            setattr(pair, key, value)
        
        mapping.touch()
        self._update("mappings", mapping)
        return pair
    
    def delete_mapping_pair(self, mapping_id: str, pair_id: str):
        mapping = self.get_mapping(mapping_id)
        if not mapping: raise ValueError("Mapping not found")

        initial_len = len(mapping.pairs)
        mapping.pairs = [p for p in mapping.pairs if p.id != pair_id]
        
        if len(mapping.pairs) == initial_len: raise ValueError("Mapping Pair not found")

        mapping.touch()
        self._update("mappings", mapping)
        return {"status": "success", "id": pair_id, "message": "Mapping Pair deleted."}


    # ---- Applications ----
    def create_application(
        self, job_id: str, base_cv_id: str,
        mapping_id: Optional[str] = None, derived_cv_id: Optional[str] = None
    ):
        app = Application.create(job_id=job_id, base_cv_id=base_cv_id, mapping_id=mapping_id, derived_cv_id=derived_cv_id)
        return self._insert("applications", app)

    def update_application(self, app_id: str, update_data: ApplicationUpdate):
        return self._update_entity("applications", Application, app_id, update_data.model_dump())

    def delete_application(self, app_id: str):
        # NOTE: In a real app, deleting an application should clean up related interviews, work items, cover letters, etc.
        return self._delete("applications", app_id)

    def get_application(self, app_id: str):
        return self._get("applications", Application, app_id)

    def all_applications(self):
        return self._all("applications", Application)


    # ---- Cover Letters ----
    def create_cover_letter(self, job_id: str, base_cv_id: str, mapping_id: str):
        cover = CoverLetter.create(job_id=job_id, base_cv_id=base_cv_id, mapping_id=mapping_id)
        return self._insert("coverletters", cover)

    def delete_cover_letter(self, cover_id: str):
        return self._delete("coverletters", cover_id)

    def get_cover_letter(self, cover_id: str):
        return self._get("coverletters", CoverLetter, cover_id)
    
    # --- NESTED ADD METHODS ---

    def add_cover_letter_idea(self, cover_id: str, title: str, description: Optional[str] = None, mapping_pair_ids: List[str] = []) -> Idea:
        cover = self.get_cover_letter(cover_id)
        if not cover:
            raise ValueError("CoverLetter not found")
        
        idea = Idea.create(title=title, description=description, mapping_pair_ids=mapping_pair_ids)
        cover.ideas.append(idea)
        cover.touch()
        self._update("coverletters", cover)
        return idea
    
    def add_cover_letter_paragraph(self, cover_id: str, idea_ids: List[str], purpose: str, draft_text: Optional[str] = None) -> Paragraph:
        cover = self.get_cover_letter(cover_id)
        if not cover:
            raise ValueError("CoverLetter not found")
        
        para = cover.add_paragraph(idea_ids=idea_ids, purpose=purpose)
        para.draft_text = draft_text # Optionally set draft text
        self._update("coverletters", cover)
        return para
    

    # --- COVER LETTER NESTED CRUD ---
    def update_cover_letter_idea(self, cover_id: str, idea_id: str, update_data: IdeaUpdate):
        cover = self.get_cover_letter(cover_id)
        if not cover: raise ValueError("CoverLetter not found")

        idea = next((i for i in cover.ideas if i.id == idea_id), None)
        if not idea: raise ValueError("Idea not found")
        
        to_update = update_data.model_dump(exclude_none=True)
        for key, value in to_update.items():
            setattr(idea, key, value)
        
        cover.touch()
        self._update("coverletters", cover)
        return idea

    def delete_cover_letter_idea(self, cover_id: str, idea_id: str):
        cover = self.get_cover_letter(cover_id)
        if not cover: raise ValueError("CoverLetter not found")

        initial_len = len(cover.ideas)
        cover.ideas = [i for i in cover.ideas if i.id != idea_id]
        
        if len(cover.ideas) == initial_len: raise ValueError("Idea not found")

        cover.touch()
        self._update("coverletters", cover)
        return {"status": "success", "id": idea_id, "message": "Idea deleted."}

    def update_cover_letter_paragraph(self, cover_id: str, para_id: str, update_data: ParagraphUpdate):
        cover = self.get_cover_letter(cover_id)
        if not cover: raise ValueError("CoverLetter not found")

        paragraph = next((p for p in cover.paragraphs if p.id == para_id), None)
        if not paragraph: raise ValueError("Paragraph not found")
        
        to_update = update_data.model_dump(exclude_none=True)
        for key, value in to_update.items():
            setattr(paragraph, key, value)
        
        cover.touch()
        self._update("coverletters", cover)
        return paragraph

    def delete_cover_letter_paragraph(self, cover_id: str, para_id: str):
        cover = self.get_cover_letter(cover_id)
        if not cover: raise ValueError("CoverLetter not found")

        initial_len = len(cover.paragraphs)
        cover.paragraphs = [p for p in cover.paragraphs if p.id != para_id]
        
        if len(cover.paragraphs) == initial_len: raise ValueError("Paragraph not found")

        cover.touch()
        self._update("coverletters", cover)
        return {"status": "success", "id": para_id, "message": "Paragraph deleted."}

    # ---- Interviews ----
    def create_interview(self, application_id: str):
        interview = Interview.create(application_id=application_id)
        return self._insert("interviews", interview)

    def delete_interview(self, interview_id: str):
        return self._delete("interviews", interview_id)

    def add_interview_stage(self, interview_id: str, name: str, description: Optional[str] = None):
        interview = self.get_interview(interview_id)
        if not interview:
            raise ValueError("Interview not found")
        # Stage creation/addition logic remains here (as defined in original models)
        stage = InterviewStage.create(name=name, description=description)
        interview.stages.append(stage)
        interview.current_stage = name
        self._update("interviews", interview)
        return stage

    def add_interview_question(self, interview_id: str, stage_name: str, question: str, answer: Optional[str] = None):
        interview = self.get_interview(interview_id)
        if not interview:
            raise ValueError("Interview not found")
        stage = next((s for s in interview.stages if s.name == stage_name), None)
        if not stage:
            raise ValueError("Stage not found")
        # Question creation/addition logic remains here (as defined in original models)
        q = InterviewQuestion.create(question=question, answer=answer)
        stage.questions.append(q)
        self._update("interviews", interview)
        return q

    def get_interview(self, interview_id: str):
        return self._get("interviews", Interview, interview_id)
    
    # --- INTERVIEW NESTED CRUD ---
    def update_interview_stage(self, interview_id: str, stage_id: str, update_data: InterviewStageUpdate):
        interview = self.get_interview(interview_id)
        if not interview: raise ValueError("Interview not found")

        stage = next((s for s in interview.stages if s.id == stage_id), None)
        if not stage: raise ValueError("Interview Stage not found")
        
        to_update = update_data.model_dump(exclude_none=True)
        for key, value in to_update.items():
            setattr(stage, key, value)
        
        interview.touch()
        self._update("interviews", interview)
        return stage

    def update_interview_question(self, interview_id: str, question_id: str, update_data: InterviewQuestionUpdate):
        interview = self.get_interview(interview_id)
        if not interview: raise ValueError("Interview not found")

        question_found = False
        for stage in interview.stages:
            question = next((q for q in stage.questions if q.id == question_id), None)
            if question:
                to_update = update_data.model_dump(exclude_none=True)
                for key, value in to_update.items():
                    setattr(question, key, value)
                question_found = True
                break
        
        if not question_found: raise ValueError("Interview Question not found")

        interview.touch()
        self._update("interviews", interview)
        return question

    def delete_interview_question(self, interview_id: str, question_id: str):
        interview = self.get_interview(interview_id)
        if not interview: raise ValueError("Interview not found")

        question_deleted = False
        for stage in interview.stages:
            initial_len = len(stage.questions)
            stage.questions = [q for q in stage.questions if q.id != question_id]
            if len(stage.questions) < initial_len:
                question_deleted = True
                break
        
        if not question_deleted: raise ValueError("Interview Question not found")

        interview.touch()
        self._update("interviews", interview)
        return {"status": "success", "id": question_id, "message": "Interview Question deleted."}


    # ---- Work Items ----
    # Renamed 'type' parameter in WorkItem.create calls to 'work_type' to avoid conflict in WorkItem model constructor
    def create_work_item(
        self,
        title: str,
        work_type: str = "research",
        related_application_id: Optional[str] = None,
        related_interview_id: Optional[str] = None,
        related_job_id: Optional[str] = None,
        related_goal_id: Optional[str] = None,
        effort_hours: Optional[float] = None,
        tags: List[str] = [],
        reflection: Optional[str] = None,
        outcome: Optional[str] = None,
    ):
        work = WorkItem.create(
            title=title,
            type=work_type, # Fixed parameter mapping to WorkItem model field
            related_application_id=related_application_id,
            related_interview_id=related_interview_id,
            related_job_id=related_job_id,
            related_goal_id=related_goal_id,
            effort_hours=effort_hours,
            tags=tags,
            reflection=reflection,
            outcome=outcome,
        )
        return self._insert("work_items", work)

    def update_work_item(self, work_id: str, update_data: WorkItemUpdate):
        return self._update_entity("work_items", WorkItem, work_id, update_data.model_dump())

    def delete_work_item(self, work_id: str):
        # Also remove the ID from any linked Goal
        work = self.get_work_item(work_id)
        if work and work.related_goal_id:
            goal = self.get_goal(work.related_goal_id)
            if goal:
                goal.work_item_ids = [w_id for w_id in goal.work_item_ids if w_id != work_id]
                self._update("goals", goal)
                self._update_goal_progress(goal.id) # Recalculate progress

        return self._delete("work_items", work_id)

    def get_work_item(self, work_id: str):
        return self._get("work_items", WorkItem, work_id)

    def all_work_items(self):
        return self._all("work_items", WorkItem)

    def mark_work_item_completed(self, work_id: str, reflection: Optional[str] = None):
        """Mark a WorkItem as completed and update its linked Goal's progress."""
        work_item = self.get_work_item(work_id)
        if not work_item:
            raise ValueError(f"WorkItem with ID {work_id} not found")

        work_item.mark_completed(reflection)
        self._update("work_items", work_item)

        if work_item.related_goal_id:
            self._update_goal_progress(work_item.related_goal_id)

        return work_item


    # ---- Goals ----
    def create_goal(self, title: str, description: Optional[str] = None, metric: Optional[str] = None):
        goal = Goal.create(title=title, description=description, metric=metric)
        return self._insert("goals", goal)

    def update_goal(self, goal_id: str, update_data: GoalUpdate):
        return self._update_entity("goals", Goal, goal_id, update_data.model_dump())

    def delete_goal(self, goal_id: str):
        # Remove related_goal_id from all linked WorkItems
        goal = self.get_goal(goal_id)
        if goal:
            for work_id in goal.work_item_ids:
                work = self.get_work_item(work_id)
                if work:
                    work.related_goal_id = None
                    self._update("work_items", work)
        return self._delete("goals", goal_id)

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
        work.related_goal_id = goal.id # Set FK on WorkItem
        self._update("goals", goal)
        self._update("work_items", work)

        self._update_goal_progress(goal_id)

        return goal

    def _update_goal_progress(self, goal_id: str):
        """Internal helper to recalculate goal progress based on linked work items."""
        goal = self.get_goal(goal_id)
        if not goal:
            return

        linked_work_items: List[WorkItem] = []
        for work_id in goal.work_item_ids:
            work_item = self.get_work_item(work_id)
            if work_item:
                linked_work_items.append(work_item)

        total_work = len(linked_work_items)
        completed_work = len([w for w in linked_work_items if w.status == "completed"])

        goal.update_progress(completed_work, total_work)
        self._update("goals", goal)

        return goal


    # ---- AI Prompt Generation Logic ----
    def generate_cv_prompt(self, base_cv_id: str, job_id: str) -> AIPromptResponse:
        """Constructs a structured prompt payload for CV generation."""
        cv = self.get_cv(base_cv_id)
        job = self.get_job(job_id)
        mapping = next((m for m in self.all_mappings() if m.job_id == job_id and m.base_cv_id == base_cv_id), None)

        if not cv or not job or not mapping:
            raise ValueError("CV, Job, or related Mapping not found for prompt generation.")

        structured_payload = CVGenerationPrompt(
            instruction="You are an expert career assistant. Your task is to generate a new Derived CV by selecting and rephrasing experiences and skills from the provided Base CV that directly address the features/requirements in the Job Description, as guided by the Mapping Data.",
            job_description=job,
            base_cv=cv,
            mapping_data=mapping,
        )

        return AIPromptResponse(
            job_id=job_id,
            cv_id=base_cv_id,
            prompt_type="CV",
            structured_payload=structured_payload,
        )

    def generate_coverletter_prompt(self, mapping_id: str) -> AIPromptResponse:
        """Constructs a structured prompt payload for Cover Letter generation."""
        mapping = self.get_mapping(mapping_id)
        if not mapping:
            raise ValueError("Mapping not found.")

        job = self.get_job(mapping.job_id)
        cv = self.get_cv(mapping.base_cv_id)
        cover_letter = next((cl for cl in self._all("coverletters", CoverLetter) if cl.mapping_id == mapping_id), None)

        if not job or not cv or not cover_letter:
            raise ValueError("Job, CV, or Cover Letter Ideas not found for prompt generation.")

        structured_payload = CoverLetterGenerationPrompt(
            instruction="You are an expert copywriter. Generate a professional cover letter. Use the Cover Letter Ideas and Paragraph structure as the outline, ensuring the text powerfully connects the Base CV experiences to the Job Description requirements using the Mapping Data as evidence.",
            job_description=job,
            base_cv=cv,
            mapping_data=mapping,
            cover_letter_ideas=cover_letter.ideas,
        )

        return AIPromptResponse(
            job_id=job.id,
            cv_id=cv.id,
            prompt_type="CoverLetter",
            structured_payload=structured_payload,
        )

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
    
