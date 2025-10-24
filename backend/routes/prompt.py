from fastapi import APIRouter

router = APIRouter()

@router.post("/generate-cv-prompt")
def generate_cv_prompt(cv_id: str, job_id: str):
    # TODO: Build structured prompt JSON for AI generation
    return {"prompt": f"Generate a tailored CV for job {job_id} using CV {cv_id}"}

@router.post("/generate-coverletter-prompt")
def generate_cover_letter_prompt(mapping_id: str):
    # TODO: Build cover letter prompt JSON from mapping and ideas
    return {"prompt": f"Generate a cover letter using mapping {mapping_id}"}
