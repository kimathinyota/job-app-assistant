# backend/core/registry.py

from typing import Optional, List, Dict, Any, Tuple
from backend.core.database import TinyDBManager
from tinydb import where
from backend.core.models import *
import logging # <-- Import logging
from bs4 import BeautifulSoup

# Set up a logger for the registry
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    
    # --- ADD THIS METHOD ---
    def _update_nested_item(self, cv_id: str, list_name: str, item_id: str, update_data: dict):
        """
        Generic helper to update fields on a nested item (e.g., an Experience)
        within a CV.
        """
        log.info(f"[Registry] Updating item {item_id} in list {list_name} for CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # Find the specific item (e.g., one experience) using the helper
        item_to_update = self._get_nested_entity(cv, list_name, item_id)
        
        # Apply the updates
        for key, value in update_data.items():
            if value is not None:
                setattr(item_to_update, key, value)
        
        item_to_update.touch()
        cv.touch()
        
        # Save the entire CV
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully updated item {item_id}.")
        return item_to_update

    # --- AND ADD THIS METHOD ---
    def _delete_nested_item(self, cv_id: str, list_name: str, item_id: str):
        """
        Generic helper to remove a nested item from a list within a CV.
        This does NOT perform a cascade delete. It's used for items
        like Experiences, Projects, etc.
        """
        log.info(f"[Registry] Deleting item {item_id} from list {list_name} in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # Get the list from the CV object (e.g., cv.experiences)
        nested_list = getattr(cv, list_name, None)
        if nested_list is None:
            raise ValueError(f"CV has no attribute '{list_name}'")

        initial_count = len(nested_list)
        
        # Re-create the list excluding the item to be deleted
        new_list = [item for item in nested_list if item.id != item_id]

        if len(new_list) == initial_count:
            # Nothing was deleted, so the item wasn't found
            raise ValueError(f"Item {item_id} not found in list '{list_name}'.")

        # Set the new list back onto the CV object
        setattr(cv, list_name, new_list)
        
        # Save changes
        cv.touch()
        self._update("cvs", cv)
        
        log.info(f"[Registry] Successfully deleted item {item_id} from {list_name}.")
        # Return a simple success message
        return {"status": "success", "id": item_id, "message": f"{list_name[:-1]} deleted"}

    
    def fetch_item(self, item_id: str, item_type: str):
        """
        Retrieves a nested entity by ID and Type.
        
        ABSTRACTS DATA ACCESS:
        - Currently: Scans loaded CVs in memory.
        - Future (MongoDB): Will run a db.collection.find_one() query.
        """
        
        # 1. Normalize type to match attribute names
        # e.g., 'experience' -> 'experiences'
        type_map = {
            'experience': 'experiences',
            'project': 'projects',
            'education': 'education', # education is usually a list named 'education'
            'skill': 'skills',
            'hobby': 'hobbies'
        }
        
        target_attr = type_map.get(item_type.lower())
        if not target_attr:
            return None

        # 2. Search Logic (The part that changes with DB switch)
        # CURRENT: In-Memory Loop (TinyDB style)
        all_cvs = self.all_cvs() # Uses your existing _all method
        
        for cv in all_cvs:
            collection = getattr(cv, target_attr, [])
            for item in collection:
                if item.id == item_id:
                    return item
        
        # FUTURE MONGODB IMPLEMENTATION EXAMPLE:
        # return self.db.cvs.find_one(
        #     {f"{target_attr}.id": item_id}, 
        #     {f"{target_attr}.$": 1}
        # )
        
        return None
    

    def fetch_item_details(self, item_id: str, item_type: str) -> Optional[Dict[str, Any]]:
        """
        Fetches an item and resolves its referenced relationships 
        (Skills, Achievements, etc.) into full objects.
        """
        item = self.fetch_item(item_id, item_type)
        if not item:
            return None
            
        # Initialize the 'Context' bucket
        result = {
            "item": item,
            "skills": [],
            "achievements": [],
            "experiences": [],
            "education": [],
            "hobbies": []
        }
        
        # Helper to fetch and append unique items
        def resolve_ids(id_list, type_key, dest_list_key):
            if not id_list: return
            for ref_id in id_list:
                obj = self.fetch_item(ref_id, type_key)
                if obj:
                    # Check for duplicates
                    if not any(x.id == obj.id for x in result[dest_list_key]):
                        result[dest_list_key].append(obj)

        # A. Resolve Skills
        if hasattr(item, 'skill_ids'):
            resolve_ids(item.skill_ids, 'skill', 'skills')

        # B. Resolve Achievements
        if hasattr(item, 'achievement_ids'):
            # Note: You need to ensure 'achievement' type is handled in fetch_item
            resolve_ids(item.achievement_ids, 'achievement', 'achievements')

        # C. Resolve Project Contexts
        if item_type == 'project':
            if hasattr(item, 'related_experience_ids'):
                resolve_ids(item.related_experience_ids, 'experience', 'experiences')
            if hasattr(item, 'related_education_ids'):
                resolve_ids(item.related_education_ids, 'education', 'education')
            if hasattr(item, 'related_hobby_ids'):
                resolve_ids(item.related_hobby_ids, 'hobby', 'hobbies')
                
        return result



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

    # --- ADD THIS NEW METHOD ---
    def delete_job_feature(self, job_id: str, feature_id: str):
        """Finds a job and removes a feature from it by ID."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")
        
        initial_count = len(job.features)
        job.features = [f for f in job.features if f.id != feature_id]
        
        if len(job.features) == initial_count:
            raise ValueError("Feature not found on this job")
            
        job.touch()
        self._update("jobs", job)
        return {"status": "success", "message": "Feature deleted"}
    # --- END OF NEW METHOD ---


    # --- ADD THIS NEW METHOD ---
    def upsert_job(self, payload: JobUpsertPayload) -> JobDescription:
        """
        Creates a new job or updates an existing one (and its features)
        from a single payload.
        """
        
        # 1. Re-create the list of JobDescriptionFeature objects
        # This also generates new IDs for any features that don't have one
        processed_features = [
            JobDescriptionFeature.create(
                id=f.id or None, # Use existing ID if provided
                type=f.type, 
                description=f.description
            ) for f in payload.features
        ]

        if payload.id:
            # --- UPDATE ---
            log.info(f"[Registry] Updating job {payload.id}")
            job = self.get_job(payload.id)
            if not job:
                raise ValueError(f"Job with ID {payload.id} not found for update.")
            
            # Update all fields from payload
            job.title = payload.title
            job.company = payload.company
            job.job_url = payload.job_url
            job.application_end_date = payload.application_end_date

            job.date_closing=payload.date_closing,       # Map directly
            job.date_posted=payload.date_posted,         # Map directly
            job.date_extracted=payload.date_extracted,   # Map directly
            job.description=payload.description,         # Map directly



            job.location = payload.location
            job.salary_range = payload.salary_range
            job.notes = payload.notes
            
            # Overwrite the features list entirely
            job.features = processed_features
            job.touch()
            self._update("jobs", job)
            return job
            
        else:
            # --- CREATE ---
            log.info("[Registry] Creating new job")
            job = JobDescription.create(
                title=payload.title,
                company=payload.company,
                job_url=payload.job_url,
                application_end_date=payload.application_end_date,
                location=payload.location,
                salary_range=payload.salary_range,
                notes=payload.notes,
                features=processed_features,
                date_closing=payload.date_closing,       # Map directly
                date_posted=payload.date_posted,         # Map directly
                date_extracted=payload.date_extracted,   # Map directly
                description=payload.description,         # Map directly
            )
            return self._insert("jobs", job)

    def get_job(self, job_id: str):
        return self._get("jobs", JobDescription, job_id)

    def all_jobs(self):
        return self._all("jobs", JobDescription)


    # ---- CVs ----

    def create_cv(self, name: str, first_name: Optional[str] = None, last_name: Optional[str] = None, title: Optional[str] = None, summary: Optional[str] = None):
        cv = CV.create(name=name, first_name=first_name, last_name=last_name, title=title, summary=summary)
        return self._insert("cvs", cv)
    

    def update_cv(self, cv_id: str, update_data: CVUpdate):
        return self._update_entity("cvs", CV, cv_id, update_data.model_dump())

    def delete_cv(self, cv_id: str):
        return self._delete("cvs", cv_id)

    def get_cv(self, cv_id: str):
        return self._get("cvs", CV, cv_id)

    def all_cvs(self):
        return self._all("cvs", CV)
    
    # --- *** NEW: Complex "Service Layer" Methods *** ---

    def _resolve_skills(
        self, 
        cv: CV, 
        new_skills_direct: List[PendingSkillInput], 
        new_achievements: List[PendingAchievementInput]
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Gathers all unique skills from all sources, creates them, and returns maps.
        Returns:
            - new_skill_id_map (Dict[str, str]): Map of {skill_name.lower(): skill_id}
            - direct_new_skill_ids (List[str]): List of skill IDs from new_skills_direct
        """
        log.info("[Registry] Resolving skills...")
        skill_map: Dict[str, PendingSkillInput] = {}

        # 1. Gather skills from the Experience itself
        for skill in new_skills_direct:
            skill_map.setdefault(skill.name.lower(), skill)
        
        # 2. Gather skills from all pending achievements
        for ach in new_achievements:
            for skill in ach.new_skills:
                skill_map.setdefault(skill.name.lower(), skill)
        
        if not skill_map:
            log.info("[Registry] No new skills to create.")
            return {}, []

        log.info(f"[Registry] Found {len(skill_map)} unique new skills to create.")
        
        # 3. Create all unique skills
        new_skill_id_map: Dict[str, str] = {}
        for skill in skill_map.values():
            # Check if skill already exists in CV by name
            existing_skill = next((s for s in cv.skills if s.name.lower() == skill.name.lower()), None)
            if existing_skill:
                new_skill_id_map[skill.name.lower()] = existing_skill.id
                log.info(f"[Registry] Skill '{skill.name}' already exists with ID {existing_skill.id}.")
            else:
                new_skill = cv.add_skill(name=skill.name, category=skill.category)
                new_skill_id_map[skill.name.lower()] = new_skill.id
                log.info(f"[Registry] Created new skill '{new_skill.name}' with ID {new_skill.id}.")

        # 4. Get the list of IDs for skills added *directly* to the experience
        direct_new_skill_ids = [new_skill_id_map[s.name.lower()] for s in new_skills_direct]

        return new_skill_id_map, direct_new_skill_ids


    def _resolve_achievements(
        self,
        cv: CV,
        new_achievements_payload: List[PendingAchievementInput],
        new_skill_id_map: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, str]]:
        log.info("[Registry] Resolving achievements...")
        if not new_achievements_payload:
            log.info("[Registry] No new achievements to create.")
            return [], {}

        new_achievement_ids: List[str] = []
        original_to_new_ach_id_map: Dict[str, str] = {}

        for ach_payload in new_achievements_payload:
            # 1. Resolve skill IDs for this achievement
            new_skill_ids = [new_skill_id_map[s.name.lower()] for s in ach_payload.new_skills]
            final_ach_skill_ids = list(set(ach_payload.existing_skill_ids + new_skill_ids))

            # --- *** START MODIFICATION *** ---
            
            # 2. Check if this is an in-place update of a master achievement
            if ach_payload.original_id:
                try:
                    # Find the original achievement in the CV's master list
                    master_ach_to_update = self._get_nested_entity(cv, 'achievements', ach_payload.original_id)
                    
                    # Update it in-place
                    log.info(f"[Registry] Updating master achievement {master_ach_to_update.id} in-place.")
                    master_ach_to_update.text = ach_payload.text
                    master_ach_to_update.context = ach_payload.context or master_ach_to_update.context
                    master_ach_to_update.skill_ids = final_ach_skill_ids
                    master_ach_to_update.touch() # Update its timestamp

                    # Map the original_id to itself, so the Experience links back to it
                    original_to_new_ach_id_map[ach_payload.original_id] = master_ach_to_update.id
                    
                    # We DO NOT add it to new_achievement_ids, because it's not a new achievement
                    
                except ValueError:
                    # Fallback: Original not found, create a new one (original behavior)
                    log.warning(f"[Registry] Original achievement {ach_payload.original_id} not found. Creating new achievement.")
                    new_ach = cv.add_achievement(
                        text=ach_payload.text,
                        context=ach_payload.context,
                        skill_ids=final_ach_skill_ids
                    )
                    new_achievement_ids.append(new_ach.id)
                    original_to_new_ach_id_map[ach_payload.original_id] = new_ach.id # Map to the new one
            else:
                # --- ORIGINAL LOGIC ---
                # This is a brand-new achievement (no original_id)
                log.info(f"[Registry] Creating new achievement '{ach_payload.text[:30]}...' with skills {final_ach_skill_ids}")
                new_ach = cv.add_achievement(
                    text=ach_payload.text,
                    context=ach_payload.context,
                    skill_ids=final_ach_skill_ids
                )
                new_achievement_ids.append(new_ach.id)
                
            # --- *** END MODIFICATION *** ---

        log.info(f"[Registry] Created {len(new_achievement_ids)} new achievements and updated others.")
        return new_achievement_ids, original_to_new_ach_id_map

    def create_experience_from_payload(self, cv_id: str, payload: ExperienceComplexPayload) -> Experience:
        log.info(f"[Registry] Starting complex create for Experience in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # Step 1: Create all new skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Create all new achievements
        new_achievement_ids, _ = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists for the Experience
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        final_achievement_ids = list(set(payload.existing_achievement_ids + new_achievement_ids))

        log.info(f"[Registry] Final Experience skill_ids: {final_skill_ids}")
        log.info(f"[Registry] Final Experience achievement_ids: {final_achievement_ids}")

        # Step 4: Create the Experience
        exp = cv.add_experience(
            title=payload.title,
            company=payload.company,
            start_date=payload.start_date,
            end_date=payload.end_date,
            description=payload.description,
            skill_ids=final_skill_ids,
            achievement_ids=final_achievement_ids
        )

        # Step 5: Save the entire updated CV
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully created new Experience {exp.id}")
        return exp

    def update_experience_from_payload(self, cv_id: str, exp_id: str, payload: ExperienceComplexPayload) -> Experience:
        log.info(f"[Registry] Starting complex update for Experience {exp_id} in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        exp = self._get_nested_entity(cv, 'experiences', exp_id)
        if not exp:
            raise ValueError("Experience not found")

        # Step 1: Create all new skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Create all new achievements
        new_achievement_ids, original_to_new_map = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists for the Experience
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        
        # For achievements, we take *unmodified* + *brand new* + *newly modified*
        final_achievement_ids = list(set(
            payload.existing_achievement_ids + 
            new_achievement_ids + 
            list(original_to_new_map.values())
        ))

        log.info(f"[Registry] Final Experience skill_ids: {final_skill_ids}")
        log.info(f"[Registry] Final Experience achievement_ids: {final_achievement_ids}")

        # Step 4: Update the Experience object directly
        exp.title = payload.title
        exp.company = payload.company
        exp.start_date = payload.start_date
        exp.end_date = payload.end_date
        exp.description = payload.description
        exp.skill_ids = final_skill_ids
        exp.achievement_ids = final_achievement_ids
        exp.touch()
        
        # Step 5: Save the entire updated CV
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully updated Experience {exp.id}")
        return exp

    # --- *** NEWLY ADDED METHODS FOR EDUCATION *** ---

    def create_education_from_payload(self, cv_id: str, payload: EducationComplexPayload) -> Education:
        log.info(f"[Registry] Starting complex create for Education in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # Step 1: Create all new skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Create all new achievements
        new_achievement_ids, _ = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        final_achievement_ids = list(set(payload.existing_achievement_ids + new_achievement_ids))

        log.info(f"[Registry] Final Education skill_ids: {final_skill_ids}")
        log.info(f"[Registry] Final Education achievement_ids: {final_achievement_ids}")

        # Step 4: Create the Education (using the new CV.add_education method)
        edu = cv.add_education(
            institution=payload.institution,
            degree=payload.degree,
            field=payload.field,
            start_date=payload.start_date,
            end_date=payload.end_date,
            skill_ids=final_skill_ids,
            achievement_ids=final_achievement_ids
        )

        # Step 5: Save the entire updated CV
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully created new Education {edu.id}")
        return edu

    def update_education_from_payload(self, cv_id: str, edu_id: str, payload: EducationComplexPayload) -> Education:
        log.info(f"[Registry] Starting complex update for Education {edu_id} in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv: raise ValueError("CV not found")
        
        edu = self._get_nested_entity(cv, 'education', edu_id)
        if not edu: raise ValueError("Education not found")

        # Step 1: Resolve Skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Resolve Achievements
        new_achievement_ids, original_to_new_map = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        final_achievement_ids = list(set(
            payload.existing_achievement_ids + 
            new_achievement_ids + 
            list(original_to_new_map.values())
        ))

        log.info(f"[Registry] Final Education skill_ids: {final_skill_ids}")
        log.info(f"[Registry] Final Education achievement_ids: {final_achievement_ids}")

        # Step 4: Update the Education object directly
        edu.institution = payload.institution
        edu.degree = payload.degree
        edu.field = payload.field
        edu.start_date = payload.start_date
        edu.end_date = payload.end_date
        edu.skill_ids = final_skill_ids
        edu.achievement_ids = final_achievement_ids
        edu.touch()
        
        # Step 5: Save
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully updated Education {edu.id}")
        return edu
    
    # --- *** NEWLY ADDED METHODS FOR HOBBY *** ---

    def create_hobby_from_payload(self, cv_id: str, payload: HobbyComplexPayload) -> Hobby:
        log.info(f"[Registry] Starting complex create for Hobby in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # Step 1: Create all new skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Create all new achievements
        new_achievement_ids, _ = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        final_achievement_ids = list(set(payload.existing_achievement_ids + new_achievement_ids))

        log.info(f"[Registry] Final Hobby skill_ids: {final_skill_ids}")
        log.info(f"[Registry] Final Hobby achievement_ids: {final_achievement_ids}")

        # Step 4: Create the Hobby
        hobby = cv.add_hobby(
            name=payload.name,
            description=payload.description,
            skill_ids=final_skill_ids,
            achievement_ids=final_achievement_ids
        )

        # Step 5: Save the entire updated CV
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully created new Hobby {hobby.id}")
        return hobby

    def update_hobby_from_payload(self, cv_id: str, hobby_id: str, payload: HobbyComplexPayload) -> Hobby:
        log.info(f"[Registry] Starting complex update for Hobby {hobby_id} in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv: raise ValueError("CV not found")
        
        hobby = self._get_nested_entity(cv, 'hobbies', hobby_id)
        if not hobby: raise ValueError("Hobby not found")

        # Step 1: Resolve Skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Resolve Achievements
        new_achievement_ids, original_to_new_map = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        final_achievement_ids = list(set(
            payload.existing_achievement_ids + 
            new_achievement_ids + 
            list(original_to_new_map.values())
        ))

        log.info(f"[Registry] Final Hobby skill_ids: {final_skill_ids}")
        log.info(f"[Registry] Final Hobby achievement_ids: {final_achievement_ids}")

        # Step 4: Update the Hobby object directly
        hobby.name = payload.name
        hobby.description = payload.description
        hobby.skill_ids = final_skill_ids
        hobby.achievement_ids = final_achievement_ids
        hobby.touch()
        
        # Step 5: Save
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully updated Hobby {hobby.id}")
        return hobby

# --- *** NEWLY ADDED METHODS FOR PROJECT *** ---

    def create_project_from_payload(self, cv_id: str, payload: ProjectComplexPayload) -> Project:
        log.info(f"[Registry] Starting complex create for Project in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        

        # --- SYNC LOGIC ---
        final_exp_ids = set(payload.related_experience_ids)
        if payload.related_experience_id:
            final_exp_ids.add(payload.related_experience_id)

        final_edu_ids = set(payload.related_education_ids)
        if payload.related_education_id:
            final_edu_ids.add(payload.related_education_id)
            
        final_hobby_ids = set(payload.related_hobby_ids)
        # ------------------


        # Step 1: Create all new skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Create all new achievements
        new_achievement_ids, _ = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        final_achievement_ids = list(set(payload.existing_achievement_ids + new_achievement_ids))

        log.info(f"[Registry] Final Project skill_ids: {final_skill_ids}")
        log.info(f"[Registry] Final Project achievement_ids: {final_achievement_ids}")

        # Step 4: Create the Project
        project = cv.add_project(
            title=payload.title,
            description=payload.description,
            
            # Pass singulars (for backward compatibility in DB view)
            # related_experience_id=payload.related_experience_id,
            # related_education_id=payload.related_education_id,
            
            # Pass plurals
            related_experience_ids=list(final_exp_ids),
            related_education_ids=list(final_edu_ids),
            related_hobby_ids=list(final_hobby_ids),
            
            skill_ids=final_skill_ids,
            achievement_ids=final_achievement_ids
        )
        # Step 5: Save the entire updated CV
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully created new Project {project.id}")
        return project


    def update_project_from_payload(self, cv_id: str, project_id: str, payload: ProjectComplexPayload) -> Project:
        log.info(f"[Registry] Starting complex update for Project {project_id} in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv: raise ValueError("CV not found")
        
        project = self._get_nested_entity(cv, 'projects', project_id)
        if not project: raise ValueError("Project not found")

        # Step 1: Resolve Skills
        new_skill_id_map, direct_new_skill_ids = self._resolve_skills(
            cv, payload.new_skills, payload.new_achievements
        )

        # Step 2: Resolve Achievements
        new_achievement_ids, original_to_new_map = self._resolve_achievements(
            cv, payload.new_achievements, new_skill_id_map
        )

        # Step 3: Consolidate final ID lists
        final_skill_ids = list(set(payload.existing_skill_ids + direct_new_skill_ids))
        final_achievement_ids = list(set(
            payload.existing_achievement_ids + 
            new_achievement_ids + 
            list(original_to_new_map.values())
        ))


        # --- SYNC LOGIC ---
        final_exp_ids = set(payload.related_experience_ids)
        if payload.related_experience_id:
            final_exp_ids.add(payload.related_experience_id)

        final_edu_ids = set(payload.related_education_ids)
        if payload.related_education_id:
            final_edu_ids.add(payload.related_education_id)
            
        final_hobby_ids = set(payload.related_hobby_ids)


        # Step 4: Update the Project object directly
        project.title = payload.title
        project.description = payload.description
        
        # Pass singulars (for backward compatibility in DB view)
        if payload.related_experience_id:
            project.related_experience_id=payload.related_experience_id
        if payload.related_education_id:
            project.related_education_id=payload.related_education_id
        
        # Pass plurals
        project.related_experience_ids=list(final_exp_ids)
        project.related_education_ids=list(final_edu_ids)
        project.related_hobby_ids=list(final_hobby_ids)

        project.skill_ids = final_skill_ids
        project.achievement_ids = final_achievement_ids

        
        project.touch()
        
        # Step 5: Save
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully updated Project {project.id}")
        return project

    # --- *** END NEW METHODS *** ---

    # --- NESTED ADD METHODS (Originals, now used by helpers) ---

    def add_cv_experience(self, cv_id: str, title: str, company: str, **kwargs) -> Experience:
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        # Pass all kwargs (like start_date, description, skill_ids, achievement_ids)
        exp = cv.add_experience(title=title, company=company, **kwargs)
        
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
        
        # --- SYNC LOGIC START ---
        # Ensure lists exist in kwargs
        exp_ids = set(kwargs.get('related_experience_ids', []))
        edu_ids = set(kwargs.get('related_education_ids', []))
        hobby_ids = set(kwargs.get('related_hobby_ids', []))

        # If old singular fields are present, add them to the sets
        if kwargs.get('related_experience_id'):
            exp_ids.add(kwargs['related_experience_id'])
        if kwargs.get('related_education_id'):
            edu_ids.add(kwargs['related_education_id'])
            
        # Update kwargs with the consolidated lists
        kwargs['related_experience_ids'] = list(exp_ids)
        kwargs['related_education_ids'] = list(edu_ids)
        kwargs['related_hobby_ids'] = list(hobby_ids)
        # --- SYNC LOGIC END ---

        proj = cv.add_project(title=title, description=description, **kwargs)
        
        if skill_ids:
            proj.skill_ids = skill_ids
            
        self._update("cvs", cv)
        return proj

    def add_cv_hobby(self, cv_id: str, name: str, description: Optional[str] = None, skill_ids: Optional[List[str]] = None, achievement_ids: Optional[List[str]] = None) -> Hobby: # <-- ADD achievement_ids parameter
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        hobby = cv.add_hobby(name=name, description=description)
        
        # Set skill_ids if they were provided
        if skill_ids:
            hobby.skill_ids = skill_ids
            
        # *** NEW: Set achievement_ids if they were provided ***
        if achievement_ids:
            hobby.achievement_ids = achievement_ids
            
        self._update("cvs", cv)
        return hobby

    def add_cv_achievement(self, cv_id: str, text: str, context: Optional[str] = None, skill_ids: Optional[List[str]] = None) -> Achievement: # <-- ADD skill_ids parameter
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
        
        # Pass skill_ids via kwargs
        ach = cv.add_achievement(text=text, context=context, skill_ids=skill_ids or [])
            
        self._update("cvs", cv)
        return ach
    
    # --- *** END NEW "Service Layer" Methods *** ---


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
        """
        Deletes a skill from the master list AND unlinks it from all
        experiences, education, projects, hobbies, and achievements.
        """
        log.info(f"[Registry] Cascade delete requested for Skill {skill_id} in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # 1. Check if skill exists before doing work
        if not any(s.id == skill_id for s in cv.skills):
            raise ValueError(f"Skill {skill_id} not found.")

        # 2. Unlink from all items that can hold skills
        all_items_with_skills = (
            cv.experiences + 
            cv.education + 
            cv.projects + 
            cv.hobbies + 
            cv.achievements  # Achievements can also have skills
        )
        
        for item in all_items_with_skills:
            if hasattr(item, 'skill_ids') and skill_id in item.skill_ids:
                log.info(f"[Registry] Unlinking Skill {skill_id} from {item.__class__.__name__} {item.id}")
                # Create a new list without the deleted ID
                item.skill_ids = [sid for sid in item.skill_ids if sid != skill_id]

        # 3. Remove the skill from the master list
        cv.skills = [s for s in cv.skills if s.id != skill_id]
        
        # 4. Save all changes
        cv.touch()
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully cascade-deleted Skill {skill_id}")
        return {"status": "success", "id": skill_id, "message": "Skill deleted and unlinked."}

    def delete_cv_achievement(self, cv_id: str, ach_id: str):
        """
        Deletes an achievement from the master list AND unlinks it from all
        experiences, education, projects, and hobbies.
        """
        log.info(f"[Registry] Cascade delete requested for Achievement {ach_id} in CV {cv_id}")
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")
            
        # 1. Check if achievement exists
        if not any(a.id == ach_id for a in cv.achievements):
            raise ValueError(f"Achievement {ach_id} not found.")

        # 2. Unlink from all items that can hold achievements
        all_items_with_achievements = (
            cv.experiences + 
            cv.education + 
            cv.projects + 
            cv.hobbies
        )
        
        for item in all_items_with_achievements:
            if hasattr(item, 'achievement_ids') and ach_id in item.achievement_ids:
                log.info(f"[Registry] Unlinking Achievement {ach_id} from {item.__class__.__name__} {item.id}")
                # Create a new list without the deleted ID
                item.achievement_ids = [aid for aid in item.achievement_ids if aid != ach_id]

        # 3. Remove the achievement from the master list
        cv.achievements = [a for a in cv.achievements if a.id != ach_id]
        
        # 4. Save all changes
        cv.touch()
        self._update("cvs", cv)
        log.info(f"[Registry] Successfully cascade-deleted Achievement {ach_id}")
        return {"status": "success", "id": ach_id, "message": "Achievement deleted and unlinked."}
    
    def delete_cv_project(self, cv_id: str, proj_id: str):
        return self._delete_nested_item(cv_id, 'projects', proj_id)
    
    def delete_cv_hobby(self, cv_id: str, hobby_id: str):
        return self._delete_nested_item(cv_id, 'hobbies', hobby_id)


    # --- NESTED LINKING METHODS ---

    def _get_nested_entity(self, cv: CV, entity_list_name: str, entity_id: str) -> Union[Experience, Education, Project, Hobby, Achievement]:
        """Helper to find a specific nested entity by ID within the CV."""
        
        # --- START MODIFICATION ---
        # Normalize common singulars to the plural list name
        list_name_map = {
            "experience": "experiences",
            "project": "projects",
            "education": "education",
            "hobby": "hobbies",
            "skill": "skills",
            "achievement": "achievements"
        }
        # Use the normalized name, but default to the original if it's already plural
        normalized_list_name = list_name_map.get(entity_list_name, entity_list_name)
        # --- END MODIFICATION ---
        
        # Use the new normalized_list_name for all checks
        if normalized_list_name == 'skills':
             entity = next((e for e in cv.skills if e.id == entity_id), None)
        elif normalized_list_name == 'experiences':
             entity = next((e for e in cv.experiences if e.id == entity_id), None)
        elif normalized_list_name == 'education':
             entity = next((e for e in cv.education if e.id == entity_id), None)
        elif normalized_list_name == 'projects':
             entity = next((e for e in cv.projects if e.id == entity_id), None)
        elif normalized_list_name == 'hobbies':
             entity = next((e for e in cv.hobbies if e.id == entity_id), None)
        elif normalized_list_name == 'achievements':
             entity = next((e for e in cv.achievements if e.id == entity_id), None)
        else:
             # The error now correctly reports the *original* failing name
             raise ValueError(f"Unknown entity type: {entity_list_name}")
        
        if not entity:
            # Use normalized_list_name in the error message
            raise ValueError(f"{normalized_list_name.capitalize()} with ID {entity_id} not found in CV.")
        return entity

    def link_skill_to_entity(self, cv_id: str, entity_id: str, skill_id: str, entity_list_name: str):
        """
        Adds a skill ID to a specific nested entity's skill_ids list (Experience, Project, etc.)
        AND rolls the link up to parents (e.g. Achievement -> Project -> Experience).
        """
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        # 1. Find the target entity (Experience, Project, etc.)
        entity = self._get_nested_entity(cv, entity_list_name, entity_id)

        # 2. Check if the skill exists in the master CV list
        skill = self._get_nested_entity(cv, 'skills', skill_id)
        
        # 3. Perform the primary link (only works if entity has the SkillLinkMixin)
        if skill_id not in entity.skill_ids:
            entity.skill_ids.append(skill_id)
            cv.touch()
            
        # 4. *** HANDLE SKILL ROLL-UP ***
        
        # --- HELPER: Generic Roll-up Function ---
        # We use this to avoid repeating the "find parent, check id, append" logic
        def _roll_up_to_parent(parent_list_name, parent_id):
            try:
                parent = self._get_nested_entity(cv, parent_list_name, parent_id)
                if skill_id not in parent.skill_ids:
                    parent.skill_ids.append(skill_id)
                    cv.touch()
            except ValueError:
                pass # Parent not found, skip

        # Case A: The entity IS a Project (Roll up to its Contexts)
        if entity_list_name == 'projects':
            # 1. Roll up to Experiences
            # (Support old singular ID for backward compatibility + new list)
            if getattr(entity, 'related_experience_id', None):
                _roll_up_to_parent('experiences', entity.related_experience_id)
            for exp_id in getattr(entity, 'related_experience_ids', []):
                _roll_up_to_parent('experiences', exp_id)

            # 2. Roll up to Education
            if getattr(entity, 'related_education_id', None):
                _roll_up_to_parent('education', entity.related_education_id)
            for edu_id in getattr(entity, 'related_education_ids', []):
                _roll_up_to_parent('education', edu_id)

            # 3. Roll up to Hobbies
            for hobby_id in getattr(entity, 'related_hobby_ids', []):
                _roll_up_to_parent('hobbies', hobby_id)


        # Case B: The entity IS an Achievement (Find who owns it, then roll up)
        if entity_list_name == 'achievements':
            
            # 1. Check Experiences
            for exp in cv.experiences:
                if entity_id in exp.achievement_ids and skill_id not in exp.skill_ids:
                    exp.skill_ids.append(skill_id)
                    cv.touch()

            # 2. Check Education (*** ADDED ***)
            for edu in cv.education:
                if entity_id in edu.achievement_ids and skill_id not in edu.skill_ids:
                    edu.skill_ids.append(skill_id)
                    cv.touch()
            
            # 3. Check Hobbies (*** ADDED ***)
            for hobby in cv.hobbies:
                if entity_id in hobby.achievement_ids and skill_id not in hobby.skill_ids:
                    hobby.skill_ids.append(skill_id)
                    cv.touch()

            # 4. Check Projects (And trigger recursive roll-up)
            for proj in cv.projects:
                if entity_id in proj.achievement_ids:
                    # A. Add to Project
                    if skill_id not in proj.skill_ids:
                        proj.skill_ids.append(skill_id)
                        cv.touch()
                    
                    # B. RECURSIVE: Roll up from Project to ITS parents
                    # (Experiences)
                    if getattr(proj, 'related_experience_id', None):
                        _roll_up_to_parent('experiences', proj.related_experience_id)
                    for pid in getattr(proj, 'related_experience_ids', []):
                        _roll_up_to_parent('experiences', pid)
                    
                    # (Education)
                    if getattr(proj, 'related_education_id', None):
                        _roll_up_to_parent('education', proj.related_education_id)
                    for pid in getattr(proj, 'related_education_ids', []):
                        _roll_up_to_parent('education', pid)

                    # (Hobbies)
                    for pid in getattr(proj, 'related_hobby_ids', []):
                        _roll_up_to_parent('hobbies', pid)

        # 5. Save all changes
        self._update("cvs", cv)
        
        return entity


    def unlink_skill_from_entity(self, cv_id: str, entity_id: str, skill_id: str, entity_list_name: str):
        """Removes a skill ID from a specific nested entity's skill_ids list."""
        # NOTE: This does NOT automatically unlink from parents, as the skill
        # might still be required by other children of that parent.
        # Automatic unlinking is much more complex and dangerous.
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

    def get_aggregated_skills_for_entity(self, cv_id: str, entity_list_name: str, entity_id: str) -> List[Skill]:
        """
        Fetches all unique skills linked to an entity AND all of its children.
        e.g., for Experience: returns skills from Experience, its Projects, and all Achievements.
        """
        cv = self.get_cv(cv_id)
        if not cv:
            raise ValueError("CV not found")

        entity = self._get_nested_entity(cv, entity_list_name, entity_id)

        total_skill_ids = set()
        if hasattr(entity, 'skill_ids'):
            total_skill_ids.update(entity.skill_ids)

        ach_ids_to_check = []
        
        # 1. Collect children IDs from the main entity
        if hasattr(entity, 'achievement_ids'):
            ach_ids_to_check.extend(entity.achievement_ids)
        
        # 2. If the entity is an Experience, also check its child Projects
        if entity_list_name == 'experiences':
            related_projects = [p for p in cv.projects if p.related_experience_id == entity_id]
            for proj in related_projects:
                if hasattr(proj, 'skill_ids'):
                    total_skill_ids.update(proj.skill_ids)
                if hasattr(proj, 'achievement_ids'):
                    ach_ids_to_check.extend(proj.achievement_ids)

        # 3. Now, check all collected achievement IDs (deduplicated)
        for ach_id in set(ach_ids_to_check):
            try:
                achievement = self._get_nested_entity(cv, 'achievements', ach_id)
                if achievement and hasattr(achievement, 'skill_ids'):
                    total_skill_ids.update(achievement.skill_ids)
            except ValueError:
                pass # Achievement not found, skip

        # 4. Resolve all unique skill IDs to Skill objects
        skill_map = {s.id: s for s in cv.skills}
        return [skill_map[sid] for sid in total_skill_ids if sid in skill_map]

    # ---- Mappings ----
    def create_mapping(self, job_id: str, base_cv_id: str):
        mapping = Mapping.create(job_id=job_id, base_cv_id=base_cv_id)
        return self._insert("mappings", mapping)

    def update_mapping(self, mapping_id: str, update_data: MappingUpdate):
        return self._update_entity("mappings", Mapping, mapping_id, update_data.model_dump())

    # --- ADD THIS METHOD to the 'Mappings' section ---
    def save_mapping(self, mapping: Mapping) -> Mapping:
        """
        Persists a fully modified Mapping object to the database.
        Crucial for the MappingOptimizer which modifies the 'pairs' list directly.
        """
        mapping.touch()
        self._update("mappings", mapping)
        return mapping

    def delete_mapping(self, mapping_id: str):
        return self._delete("mappings", mapping_id)

# --- THIS IS THE KEY LOGIC CHANGE ---
    def add_mapping_pair(
        self, 
        mapping_id: str, 
        feature: JobDescriptionFeature, 
        context_item: Union[Experience, Project, Education, Hobby],
        context_item_type: str,
        annotation: Optional[str] = None
    ):
        mapping = self.get_mapping(mapping_id)
        if not mapping:
            raise ValueError("Mapping not found")

        # Check for duplicates (if no annotation is provided)
        if not annotation:
            for existing_pair in mapping.pairs:
                if existing_pair.feature_id == feature.id and existing_pair.context_item_id == context_item.id:
                    raise ValueError("This pair already exists. Add an annotation to create a new link.")

        # --- THIS IS THE CRITICAL PART ---
        # Get the display text for the context item
        if context_item_type == 'experiences':
            item_text = f"{context_item.title} @ {context_item.company}"
        elif context_item_type == 'projects':
            item_text = f"{context_item.title} (Project)"
        elif context_item_type == 'education':
            item_text = f"{context_item.degree} @ {context_item.institution}"
        elif context_item_type == 'hobbies':
            item_text = f"{context_item.name} (Hobby)"
        else:
            item_text = "Unknown Item"
            
        pair = MappingPair.create(
            feature_id=feature.id,
            context_item_id=context_item.id,
            context_item_type=context_item_type,
            feature_text=feature.description,
            context_item_text=item_text,           # <-- THIS LINE MUST BE HERE
            annotation=annotation,
        )
        # --- END CRITICAL PART ---
        
        # This appends to the MAPPING object, not the REGISTRY
        mapping.pairs.append(pair)
        mapping.touch()
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

    # --- NEW METHOD ---
    def get_app_suite_data(self) -> Dict[str, Any]:
        """Fetches all jobs and applications for the app suite view."""
        log.info("[Registry] Fetching all jobs and applications...")
        jobs = self.all_jobs()
        applications = self.all_applications()
        return {"jobs": jobs, "applications": applications}
    # --- END NEW METHOD ---

    # ---- Cover Letters ----
    def create_cover_letter(self, job_id: str, base_cv_id: str, mapping_id: str, name: str = "Cover Letter"):
        cover = CoverLetter.create(name=name, job_id=job_id, base_cv_id=base_cv_id, mapping_id=mapping_id)
        return self._insert("coverletters", cover)

    def rename_cover_letter(self, cover_id: str, new_name: str):
        cover = self.get_cover_letter(cover_id)
        if not cover: raise ValueError("Document not found")
        cover.name = new_name
        cover.touch()
        self._update("coverletters", cover)
        return cover
    
    def add_document_to_application(self, app_id: str, doc_id: str):
        app = self.get_application(app_id)
        if doc_id not in app.supporting_document_ids:
            app.supporting_document_ids.append(doc_id)
            self._update("applications", app)
        return app
    
    

    def submit_application(self, app_id: str):
        """
        Promotes an application to 'Applied' and locks all related assets
        to create a pseudo-snapshot.
        """
        app = self.get_application(app_id)
        if not app: raise ValueError("Application not found")

        # 1. Lock the Application
        app.status = "applied"
        app.is_locked = True
        app.applied_at = datetime.utcnow()
        
        # 2. Lock the Mapping
        if app.mapping_id:
            mapping = self.get_mapping(app.mapping_id)
            if mapping:
                mapping.is_locked = True
                self._update("mappings", mapping)

        # 3. Lock the CV
        if app.derived_cv_id:
            cv = self.get_cv(app.derived_cv_id)
            if cv: # Assuming DerivedCV
                cv.is_locked = True
                self._update("cvs", cv)

        # 4. Lock all Supporting Docs
        for doc_id in app.supporting_document_ids:
            doc = self.get_cover_letter(doc_id)
            if doc:
                doc.is_locked = True
                self._update("coverletters", doc)

        self._update("applications", app)
        return app
    

    
    def delete_cover_letter(self, cover_id: str):
        return self._delete("coverletters", cover_id)

    def get_cover_letter(self, cover_id: str):
        return self._get("coverletters", CoverLetter, cover_id)
    
    # --- NESTED ADD METHODS ---

    def add_cover_letter_idea(self, cover_id: str, title: str, description: Optional[str] = None, mapping_pair_ids: List[str] = [], annotation: Optional[str] = None) -> Idea:
        cover = self.get_cover_letter(cover_id)
        if not cover:
            raise ValueError("CoverLetter not found")
        
        idea = Idea.create(title=title, description=description, mapping_pair_ids=mapping_pair_ids)
        cover.ideas.append(idea)
        cover.touch()
        self._update("coverletters", cover)
        return idea
    
# --- UPDATED METHOD ---
    def add_cover_letter_paragraph(self, cover_id: str, idea_ids: List[str], purpose: str, draft_text: Optional[str] = None, order: Optional[int] = None) -> Paragraph:
        cover = self.get_cover_letter(cover_id)
        if not cover:
            raise ValueError("CoverLetter not found")
        
        # 1. Create the new paragraph object
        # We initially give it 0; the re-indexing step below will set the real order.
        new_para = Paragraph.create(order=0, idea_ids=idea_ids, purpose=purpose)
        new_para.draft_text = draft_text 

        # 2. Sort existing paragraphs to ensure our insertion index aligns with the visual order
        #    (This fixes issues where the DB list might be out of sync with 'order' fields)
        cover.paragraphs.sort(key=lambda x: x.order if x.order is not None else 999)

        # 3. Insert (or Append) the new paragraph at the specific index
        if order is not None and 0 <= order <= len(cover.paragraphs):
            cover.paragraphs.insert(order, new_para)
        else:
            cover.paragraphs.append(new_para)
        
        # 4. Robust Re-indexing: Reset 'order' for the entire list to 0, 1, 2...
        #    This guarantees no duplicates, gaps, or shift errors.
        for idx, para in enumerate(cover.paragraphs):
            para.order = idx
        
        cover.touch()
        self._update("coverletters", cover)
        
        # The new_para now has the correct 'order' set by the loop above
        return new_para

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

    def autofill_cover_letter(self, cover_id: str, strategy: str = "standard", mode: str = "reset"):
        """
        Intelligently populates a cover letter with a theme-based, evidence-driven outline.
        Implements the "Ownership & Re-Classification Engine"
        
        This is a non-destructive operation that:
        1. Preserves all 'user' owned Paragraphs and Ideas.
        2. Re-classifies 'user' Ideas into the new structure.
        3. Prevents duplicate AI Ideas by checking for 'user' claimed evidence.
        
        NOTE: This logic assumes 'owner: Literal["user", "autofill"]'
        and 'classification: Literal["professional", "personal", "company", "unclassified"]'
        have been added to the Idea model.
        And 'owner' has been added to the Paragraph model.
        """
        cover = self.get_cover_letter(cover_id)
        if not cover:
            raise ValueError("Cover letter not found")

        # 1. FETCH CONTEXT
        mapping = self.get_mapping(cover.mapping_id)
        job = self.get_job(cover.job_id)
        if not mapping or not job:
            raise ValueError("Missing mapping or job data")

        features_map = {f.id: f for f in job.features}
        
        # ------------------------------------------------------------------
        # STEP 1: FIND ALL "SURVIVORS" (USER-OWNED CONTENT)
        # ------------------------------------------------------------------
        
        survivor_paras = [p for p in cover.paragraphs if getattr(p, 'owner', 'user') == "user"]
        survivor_ideas = [i for i in cover.ideas if getattr(i, 'owner', 'user') == "user"]

        # ------------------------------------------------------------------
        # STEP 2: FIND "AVAILABLE EVIDENCE" (Prevents Duplicates)
        # ------------------------------------------------------------------
        
        claimed_pair_ids = set()
        for idea in survivor_ideas:
            claimed_pair_ids.update(idea.mapping_pair_ids)

        all_mapping_pairs = mapping.pairs
        
        available_pairs_for_ai = [
            pair for pair in all_mapping_pairs
            if pair.id not in claimed_pair_ids and pair.feature_id in features_map
        ]

        # ------------------------------------------------------------------
        # STEP 3: RUN AI CLUSTERING (On "Leftover" Evidence)
        # ------------------------------------------------------------------
        
        ai_themed_clusters = {}
        for pair in available_pairs_for_ai:
            ai_themed_clusters.setdefault(pair.feature_id, []).append(pair.id)
        
        new_ai_ideas = []
        for feature_id, pair_ids in ai_themed_clusters.items():
            feature = features_map[feature_id]
            new_idea = Idea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                owner="autofill", 
                title=feature.description,
                mapping_pair_ids=pair_ids,
                classification="unclassified" 
            )
            
            # --- Classify the new AI idea ---
            first_pair = next((p for p in all_mapping_pairs if p.id == pair_ids[0]), None)
            context_type = getattr(first_pair, 'context_item_type', None)

            if feature.type in ["value", "benefit"]:
                new_idea.classification = "company"
            elif context_type == "hobbies" or feature.type == "nice_to_have":
                new_idea.classification = "personal"
            elif feature.type in ["requirement", "qualification", "responsibility"]:
                new_idea.classification = "professional"
            
            new_ai_ideas.append(new_idea)

        # ------------------------------------------------------------------
        # STEP 4: RE-CLASSIFY & MERGE ALL IDEAS (The "Engine")
        # ------------------------------------------------------------------

        final_pro_ideas = []
        final_per_ideas = []
        final_com_ideas = []

        def classify_idea(idea: Idea) -> str:
            classification = getattr(idea, 'classification', 'unclassified')
            if classification != 'unclassified':
                return classification

            if idea.mapping_pair_ids:
                first_pair_id = idea.mapping_pair_ids[0]
                pair = next((p for p in all_mapping_pairs if p.id == first_pair_id), None)
                if pair and pair.feature_id in features_map:
                    feature = features_map[pair.feature_id]
                    context_type = getattr(pair, 'context_item_type', None)

                    if feature.type in ["value", "benefit"]:
                        return "company"
                    if context_type == "hobbies" or feature.type == "nice_to_have":
                        return "personal"
                    if feature.type in ["requirement", "qualification", "responsibility"]:
                        return "professional"
            
            return "professional" 

        all_ideas_to_place = survivor_ideas + new_ai_ideas

        for idea in all_ideas_to_place:
            bucket_key = classify_idea(idea)
            
            if bucket_key == "company":
                final_com_ideas.append(idea)
            elif bucket_key == "personal":
                final_per_ideas.append(idea)
            else:
                final_pro_ideas.append(idea)

        # 4. HANDLE THE "RESEARCH GAP"
        if not final_com_ideas:
            mapped_feature_ids = set(pair.feature_id for pair in all_mapping_pairs)
            unmapped_values = [f.description for f in job.features if f.type == "value" and f.id not in mapped_feature_ids]
            unmapped_benefits = [f.description for f in job.features if f.type == "benefit" and f.id not in mapped_feature_ids]
            annotation_text = "Click 'Edit' to add your research.\n\n"
            if unmapped_values: annotation_text += "Unmapped Company Values to consider:\n" + "\n".join(f"* {v}" for v in unmapped_values) + "\n\n"
            if unmapped_benefits: annotation_text += "Culture Clues (from Benefits) to consider:\n" + "\n".join(f"* {b}" for b in unmapped_benefits) + "\n\n"
            annotation_text += "Research Links:\n"
            annotation_text += f"* [Google News: {job.company}](https://www.google.com/search?q={job.company.replace(' ', '+')}+news)\n"
            if job.job_url:
                annotation_text += f"* [Company Website]({job.job_url})\n"

            
            gap_idea = Idea(
                id=f"idea_{uuid.uuid4().hex[:8]}",
                owner="autofill", 
                title=f"❓ Why {job.company}?",
                mapping_pair_ids=[], 
                annotation=annotation_text,
                classification="company" 
            )
            final_com_ideas.append(gap_idea)
            new_ai_ideas.append(gap_idea) 

        # ------------------------------------------------------------------
        # STEP 5: ASSEMBLE FINAL OUTLINE (THE "NON-DESTRUCTIVE" MERGE)
        # ------------------------------------------------------------------
        
        # A) Delete *only* the old AI-generated content
        ai_para_ids_to_delete = [p.id for p in cover.paragraphs if getattr(p, 'owner', 'user') == "autofill"]
        ai_idea_ids_to_delete = [i.id for i in cover.ideas if getattr(i, 'owner', 'user') == "autofill"]
        
        cover.paragraphs = [p for p in cover.paragraphs if p.id not in ai_para_ids_to_delete]
        cover.ideas = [i for i in cover.ideas if i.id not in ai_idea_ids_to_delete]

        # B) Add all *new* AI Ideas to the master list
        cover.ideas.extend(new_ai_ideas)

        # C) Define the new AI Paragraph "Buckets"
        pro_para_purpose = "My Professional Fit (The 'Why Me')"
        per_para_purpose = "My Personal Fit (The 'Differentiator')"
        com_para_purpose = f"My Alignment with {job.company} (The 'Why You')"
        
        new_ai_paragraphs_map = {}
        
        if final_pro_ideas:
            new_ai_paragraphs_map[pro_para_purpose] = Paragraph(
                id=f"para_{uuid.uuid4().hex[:8]}", owner="autofill", 
                purpose=pro_para_purpose, idea_ids=[i.id for i in final_pro_ideas],
                order=0 
            )
        if final_per_ideas:
            new_ai_paragraphs_map[per_para_purpose] = Paragraph(
                id=f"para_{uuid.uuid4().hex[:8]}", owner="autofill", 
                purpose=per_para_purpose, idea_ids=[i.id for i in final_per_ideas],
                order=0
            )
        if final_com_ideas:
            new_ai_paragraphs_map[com_para_purpose] = Paragraph(
                id=f"para_{uuid.uuid4().hex[:8]}", owner="autofill", 
                purpose=com_para_purpose, idea_ids=[i.id for i in final_com_ideas],
                order=0
            )

        # --- D) Build the final paragraph list based on strategy ---
        final_paragraphs = []
        
        # This index will be assigned to every paragraph to ensure
        # the 'order' field is always populated.
        order_index = 0 
        
        # Add Intro (survivor or new)
        intro_para = next((p for p in survivor_paras if p.purpose == "Introduction"), None)
        if not intro_para:
            intro_para = Paragraph(
                id=f"para_{uuid.uuid4().hex[:8]}", 
                owner="autofill", 
                purpose="Introduction", 
                idea_ids=[],
                order=order_index # Assign order
            )
        else:
            intro_para.order = order_index # Re-order survivor
        final_paragraphs.append(intro_para)
        order_index += 1

        # --- Assemble order based on strategy ---
        strategy_order_map = {
            "standard": [pro_para_purpose, per_para_purpose, com_para_purpose],
            "mission_driven": [com_para_purpose, pro_para_purpose, per_para_purpose],
            "specialist": [pro_para_purpose, per_para_purpose]
        }
        
        standard_purposes = {pro_para_purpose, per_para_purpose, com_para_purpose, "Introduction", "Conclusion"}

        for purpose_name in strategy_order_map.get(strategy, "standard"):
            survivor = next((p for p in survivor_paras if p.purpose == purpose_name), None)
            
            if survivor:
                survivor.order = order_index # Re-order survivor
                final_paragraphs.append(survivor)
                order_index += 1
            else:
                new_ai_para = new_ai_paragraphs_map.get(purpose_name)
                if new_ai_para:
                    new_ai_para.order = order_index # Assign order
                    final_paragraphs.append(new_ai_para)
                    order_index += 1
        
        # Append any *other* custom user paragraphs
        for p in survivor_paras:
            if p.purpose not in standard_purposes:
                p.order = order_index # Assign order
                final_paragraphs.append(p)
                order_index += 1

        # Add Conclusion (survivor or new)
        outro_para = next((p for p in survivor_paras if p.purpose == "Conclusion"), None)
        if not outro_para:
            outro_para = Paragraph(
                id=f"para_{uuid.uuid4().hex[:8]}", 
                owner="autofill", 
                purpose="Conclusion", 
                idea_ids=[],
                order=order_index # Assign order
            )
        else:
            outro_para.order = order_index # Re-order survivor
        final_paragraphs.append(outro_para)
        
        # Set the final, merged list of paragraphs
        cover.paragraphs = final_paragraphs
        
        # ------------------------------------------------------------------
        # ******** THIS IS THE FIX for the 'save' and 'db' errors *********
        #
        # We modified the 'cover' object in memory. Now we must use the
        # registry's internal _update method (which you provided)
        # to persist these changes.
        self._update("coverletters", cover)
        #
        # ******** END OF FIX *********
        
        return self.get_cover_letter(cover_id)
    
    # ... (rest of your registry.py methods)
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
    # ---- AI Prompt Generation Logic ----
    def generate_cv_prompt(
        self, 
        base_cv_id: str, 
        job_id: str, 
        selected_skill_ids: Optional[List[str]] # <-- 1. Add parameter
    ) -> AIPromptResponse:
        """Constructs a structured prompt payload for CV generation."""
        cv = self.get_cv(base_cv_id)
        job = self.get_job(job_id)
        mapping = next((m for m in self.all_mappings() if m.job_id == job_id and m.base_cv_id == base_cv_id), None)

        log.info(f"[Registry][P1] Generated CV prompt for Job ID {job.id} and CV ID {cv.id} using Mapping ID {mapping.id}.")


        if not cv or not job or not mapping:
            log.warning(f"DEBUG: cv={cv}, job={job}, mapping={mapping}")
            raise ValueError("CV, Job, or related Mapping not found for prompt generation.")

        log.info(f"[Registry][P2] Generated CV prompt for Job ID {job.id} and CV ID {cv.id} using Mapping ID {mapping.id}.")
        # --- 2. Create a copy of the CV to filter ---
        cv_for_prompt = cv.model_copy(deep=True)

        # --- 3. Filter the skills based on the provided list ---
        # We also need to find all mapped items to pass to the prompt
        
        mapped_item_ids = {p.context_item_id or p.experience_id for p in mapping.pairs}
        
        cv_for_prompt.experiences = [e for e in cv_for_prompt.experiences if e.id in mapped_item_ids]
        cv_for_prompt.education = [e for e in cv_for_prompt.education if e.id in mapped_item_ids]
        cv_for_prompt.projects = [p for p in cv_for_prompt.projects if p.id in mapped_item_ids]
        cv_for_prompt.hobbies = [h for h in cv_for_prompt.hobbies if h.id in mapped_item_ids]

        if selected_skill_ids is not None:
            log.info(f"[Registry] Filtering CV skills to {len(selected_skill_ids)} selected IDs.")
            cv_for_prompt.skills = [s for s in cv_for_prompt.skills if s.id in selected_skill_ids]
        else:
            # Fallback: if no list is provided, just use all skills
            log.info("[Registry] No skill filter provided, using all skills.")
            pass # cv_for_prompt.skills already contains all skills

        log.info(f"[Registry][P3] Generated CV prompt for Job ID {job.id} and CV ID {cv.id} using Mapping ID {mapping.id}.")

        structured_payload = CVGenerationPrompt(
            instruction="You are an expert career assistant. Your task is to generate a new Derived CV by selecting and rephrasing experiences and skills from the provided Base CV that directly address the features/requirements in the Job Description, as guided by the Mapping Data.",
            job_description=job,
            base_cv=cv_for_prompt, 
            mapping_data=mapping,
            goal="Generate a derived CV by prioritizing experience in the CV that is linked to job requirements in the mapping_data." # <-- ADD THIS LINE
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
        
        log.info(f"[Registry] [p2] Generated Cover Letter prompt for Job ID {job.id} and CV ID {cv.id} using Mapping ID {mapping.id}.")

        structured_payload = CoverLetterGenerationPrompt(
            instruction="You are an expert copywriter. Generate a professional cover letter. Use the Cover Letter Ideas and Paragraph structure as the outline, ensuring the text powerfully connects the Base CV experiences to the Job Description requirements using the Mapping Data as evidence.",
            job_description=job,
            base_cv=cv,
            mapping_data=mapping,
            cover_letter_ideas=cover_letter.ideas,
        )

        log.info(f"[Registry] [p3] Generated Cover Letter prompt for Job ID {job.id} and CV ID {cv.id} using Mapping ID {mapping.id}.")

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
    

    def _parse_codex_html(self, html_content: str, reference_accumulator: Dict[str, Any], cv: CV) -> List[PromptSegment]:
        """
        Parses the RichTextEditor HTML.
        Separates 'Visible Text' (Draft) from 'Context Injection' (Ghost Text).
        Extracts @mentions and auto-registers them in the bank.
        """
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        segments = []

        # --- Helper to register references found in text ---
        def register_reference(ref_id, context, label):
            if ref_id in reference_accumulator: return
            
            found_item = None
            for list_name in ['experiences', 'projects', 'education', 'skills', 'achievements', 'hobbies']:
                try:
                    found_item = self._get_nested_entity(cv, list_name, ref_id)
                    if found_item: break
                except: continue
            
            detail_text = "(Detail not found)"
            if found_item:
                d_list = []
                if hasattr(found_item, 'description') and found_item.description: d_list.append(found_item.description)
                if hasattr(found_item, 'text') and found_item.text: d_list.append(found_item.text)
                if hasattr(found_item, 'title'): d_list.append(f"Role: {found_item.title}")
                if hasattr(found_item, 'company'): d_list.append(f"Company: {found_item.company}")
                detail_text = "\n".join(d_list)

            reference_accumulator[ref_id] = PromptReferenceItem(
                id=ref_id, type="cv_item", name=label, detail=detail_text
            )

        for element in soup.recursiveChildGenerator():
            # 1. Plain Text
            if element.name is None: 
                text = element.strip()
                if text:
                    parent = element.parent
                    is_ghost = parent and parent.has_attr("class") and "ghost-text-mark" in parent["class"]
                    segments.append(PromptSegment(
                        type="context_injection" if is_ghost else "text", 
                        content=f"User Thought: {text}" if is_ghost else text
                    ))
            
            # 2. Mention Chips
            elif element.name == "span" and "mention-chip" in (element.get("class") or []):
                is_ghost = element.get("isghost") == "true" or element.get("data-ghost") == "true"
                ref_id = element.get("id")
                label = element.get("label") or element.get_text()
                
                if ref_id: register_reference(ref_id, element.get("context"), label)
                
                # Ghost references are instructions ("Talk about X"). Visible are text ("I worked at X").
                content_str = f"{label} (ID: {ref_id})"
                segments.append(PromptSegment(
                    type="context_injection" if is_ghost else "text",
                    content=f"Reference: {content_str}" if is_ghost else content_str
                ))

            # 3. Section Titles
            elif element.name == "div" and element.get("data-type") == "section-title":
                segments.append(PromptSegment(type="text", content=f"\n## {element.get_text()}\n"))

        return segments

    def construct_advanced_cover_letter_prompt(self, cover_id: str) -> CoverLetterPromptPayload:
        """
        Builds the 'Greedy' Prompt Payload.
        It loads ALL relevant context (Mapped & Unmapped) so the AI can be fully creative.
        """
        cover = self.get_cover_letter(cover_id)
        job = self.get_job(cover.job_id)
        cv = self.get_cv(cover.base_cv_id)
        mapping = self.get_mapping(cover.mapping_id)

        if not cover or not job or not cv: raise ValueError("Missing core assets")

        reference_bank: Dict[str, PromptReferenceItem] = {}
        mapped_req_ids = set()
        mapped_ev_ids = set()

        # --- A. Helper: Add Item to Bank ---
        def add_to_bank(item_id, kind="cv_item", specific_obj=None):
            if not item_id or item_id in reference_bank: return
            
            obj = specific_obj
            if not obj:
                # Try to find if not provided
                if kind == "job_feature":
                    obj = next((f for f in job.features if f.id == item_id), None)
                else:
                    for lst in ['experiences', 'projects', 'education', 'skills', 'achievements', 'hobbies']:
                        try:
                            obj = self._get_nested_entity(cv, lst, item_id)
                            if obj: break
                        except: continue
            
            if obj:
                # Build Detail String
                details = []
                if kind == "job_feature":
                    details.append(obj.description)
                else:
                    if hasattr(obj, 'title'): details.append(f"Role: {obj.title}")
                    if hasattr(obj, 'company'): details.append(f"Company: {obj.company}")
                    if hasattr(obj, 'institution'): details.append(f"Institution: {obj.institution}")
                    if hasattr(obj, 'description') and obj.description: details.append(obj.description)
                    if hasattr(obj, 'text') and obj.text: details.append(obj.text) # Achievements
                    # Add skills if linked
                    if hasattr(obj, 'skill_ids'):
                        skill_names = [s.name for s in cv.skills if s.id in obj.skill_ids]
                        if skill_names: details.append(f"Skills: {', '.join(skill_names)}")

                reference_bank[item_id] = PromptReferenceItem(
                    id=item_id, 
                    type=kind, 
                    name=getattr(obj, 'title', getattr(obj, 'name', getattr(obj, 'description', 'Item'))), 
                    detail="\n".join(details)
                )

        # --- B. Greedy Hydration (Load EVERYTHING) ---
        
        # 1. Load ALL Job Features
        all_req_ids = []
        for feat in job.features:
            add_to_bank(feat.id, "job_feature", feat)
            all_req_ids.append(feat.id)

        # 2. Load ALL Significant CV Items (Exp, Proj, Edu)
        all_cv_ids = []
        for item in (cv.experiences + cv.projects + cv.education):
            add_to_bank(item.id, "cv_item", item)
            all_cv_ids.append(item.id)
        
        # --- C. Process Explicit Mappings ---
        available_mappings = []
        for pair in mapping.pairs:
            mapped_req_ids.add(pair.feature_id)
            mapped_ev_ids.add(pair.context_item_id)
            
            available_mappings.append(PromptEvidenceLink(
                requirement_ref=pair.feature_id,
                evidence_ref=pair.context_item_id,
                annotation=pair.annotation
            ))

        # --- D. Calculate "Unmapped Potential" ---
        unmapped_reqs = [rid for rid in all_req_ids if rid not in mapped_req_ids]
        unused_cv_items = [cid for cid in all_cv_ids if cid not in mapped_ev_ids]

        # --- E. Process Outline (Structure) ---
        prompt_paragraphs = []
        sorted_paras = sorted(cover.paragraphs, key=lambda p: p.order)

        for para in sorted_paras:
            # 1. Parse User Text
            segments = self._parse_codex_html(para.draft_text, reference_bank, cv)
            
            # 2. Process Linked Ideas (The Graph)
            para_args = []
            for idea_id in para.idea_ids:
                idea = next((i for i in cover.ideas if i.id == idea_id), None)
                if not idea: continue
                
                # Ensure explicit links are in bank (if not already)
                for eid in idea.related_entity_ids: add_to_bank(eid)
                
                evidence = []
                for pid in idea.mapping_pair_ids:
                    pair = next((p for p in mapping.pairs if p.id == pid), None)
                    if pair:
                        evidence.append(PromptEvidenceLink(
                            requirement_ref=pair.feature_id, evidence_ref=pair.context_item_id, annotation=pair.annotation
                        ))
                
                para_args.append(PromptArgument(
                    topic=idea.title, user_strategy_notes=idea.annotation, evidence_links=evidence
                ))

            prompt_paragraphs.append(PromptParagraph(
                order=para.order, purpose=para.purpose, user_draft_segments=segments, key_arguments=para_args
            ))

        # --- F. Global Instructions (UPDATED) ---
        instructions = [
            "ROLE: Expert Career Writer. You are writing a persuasive cover letter on behalf of the candidate.",
            "TASK: Write a highly tailored, narrative-driven letter that bridges the candidate's past to the company's future.",
            
            "DATA SOURCES:",
            "  1. 'reference_bank': The Dictionary. Look up ALL IDs here (e.g., 'exp_123') for full details. NEVER invent details.",
            "  2. 'outline': The Skeleton. If provided, follow this structure exactly.",
            "  3. 'available_mappings': The Strongest Links. Prioritize these connections.",
            "  4. 'unmapped_job_requirements' & 'unused_cv_items': The Hidden Gems. If a paragraph feels thin, OR if you see a connection the user missed, pull from these lists to strengthen the letter.",
            
            "EXECUTION & TONE:",
            "  - **Contextualize Everything:** Never assume the reader knows the candidate's history. When introducing an experience (e.g., 'exp_123'), explicitly state the Role, Company, and a brief context naturally (e.g., 'During my time as a Senior Engineer at TechCorp, a cloud solutions provider...'). Do not say 'As seen in my CV'.",
            "  - **Hidden STAR Framework:** You MUST use the STAR method (Situation, Task, Action, Result) to structure your arguments, but **DO NOT use labels** like 'Situation:' or 'Task:'. Write flowing, cohesive paragraphs. The STAR structure should be the *invisible skeleton* of the narrative.",
            "  - **Persuasive Flow:** Connect the 'Why Me' arguments directly to the company's needs. Don't just list skills; explain *how* those skills solve the company's specific problems (found in job_context).",
            "  - If 'user_draft_segments' has text, polish it to meet these standards while keeping the user's voice.",
            "  - If 'user_draft_segments' has 'context_injection' (ghost text), use it as a directive but do not print it.",
            "  - If a paragraph is empty, generate it entirely using the 'key_arguments' AND any relevant 'available_mappings', applying the narrative STAR framework.",
            "  - If the entire outline is empty, ARCHITECT the letter yourself using the strongest matches from 'available_mappings' and generate more of these mappings if there are not enough."
        ]

        return CoverLetterPromptPayload(
            job_context={"title": job.title, "company": job.company, "notes": job.notes},
            candidate_profile_summary={"name": f"{cv.first_name} {cv.last_name}", "title": cv.title},
            reference_bank=reference_bank,
            outline=prompt_paragraphs,
            available_mappings=available_mappings,
            unmapped_job_requirements=unmapped_reqs,
            unused_cv_items=unused_cv_items,
            global_instructions=instructions
        )
    

