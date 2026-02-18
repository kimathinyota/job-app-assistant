import axios from 'axios';

// The base URL MUST match the host/port defined in your backend/main.py
// const API_BASE_URL = 'http://192.168.1.161:8000/api'; // <-- THIS IS THE FIX

export const API_BASE_URL = "http://localhost:8000/api"

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
    headers: {
        // We generally use application/json, but the POST /cv endpoint needs special handling (see createBaseCV)
        'Content-Type': 'application/json', 
    },
});

// --- JobDescription Endpoints (routes/job.py) ---
// 1. UPDATE: Accept params for Search & Sort
export const fetchAllJobs = (params = {}) => {
    // params example: { q: 'python', sort: 'recommended' }
    const query = new URLSearchParams(params).toString();
    return apiClient.get(`/job/?${query}`);
};

export const scoreAllJobs = (cvId) => {
    // If cvId is null, backend uses user.primary_cv_id
    const params = cvId ? `?cv_id=${cvId}` : '';
    return apiClient.post(`/job/score-all${params}`);
};

export const fetchJobDetails = (jobId) => apiClient.get(`/job/${jobId}`);
export const createJob = (title, company) => {
    const params = new URLSearchParams({ title, company });
    return apiClient.post(`/job/?${params.toString()}`);
};
export const addJobFeature = (jobId, description, type = 'requirement') => {
    const params = new URLSearchParams({ description, type });
    return apiClient.post(`/job/${jobId}/feature?${params.toString()}`);
};

// --- Application Endpoints (routes/application.py) ---
export const fetchAllApplications = () => apiClient.get('/application/');
export const fetchApplicationDetails = (appId) => apiClient.get(`/application/${appId}`);
export const createApplication = (jobId, baseCvId, mappingId) => {
    const params = new URLSearchParams({ 
        job_id: jobId, 
        base_cv_id: baseCvId, 
        mapping_id: mappingId 
    });
    return apiClient.post(`/application/?${params.toString()}`);
};
export const updateApplicationStatus = (appId, status) => {
    // Uses PUT /application/{app_id}/status
    return apiClient.put(`/application/${appId}/status`, { status });
};

// --- NEW: General application update function ---
export const updateApplication = (appId, updateData) => {
    // updateData is an object, e.g., { cover_letter_id: "..." }
    return apiClient.patch(`/application/${appId}`, updateData);
};




// --- Mapping Endpoints (routes/mapping.py) ---
export const fetchMappingDetails = (mappingId) => apiClient.get(`/mapping/${mappingId}`);
export const createMapping = (jobId, baseCvId) => {
    const params = new URLSearchParams({ job_id: jobId, base_cv_id: baseCvId });
    return apiClient.post(`/mapping/?${params.toString()}`);
};


export const addMappingPair = (mappingId, featureId, contextItemId, contextItemType, annotation) => {
    const params = new URLSearchParams({ 
        feature_id: featureId, 
        context_item_id: contextItemId,
        context_item_type: contextItemType
    });
    
    // Add annotation only if it exists
    if (annotation) {
        params.append('annotation', annotation);
    }
    
    return apiClient.post(`/mapping/${mappingId}/pair?${params.toString()}`);
};

export const deleteMappingPair = (mappingId, pairId) => {
    // This function was being called by the UI but didn't exist here
    return apiClient.delete(`/mapping/${mappingId}/pair/${pairId}`);
};




// --- CoverLetter Endpoints (routes/coverletter.py) ---
export const createCoverLetter = (jobId, baseCvId, mappingId) => {
     const params = new URLSearchParams({ 
        job_id: jobId, 
        base_cv_id: baseCvId, 
        mapping_id: mappingId 
    });

    return apiClient.post(`/coverletter/?${params.toString()}`);
};

// --- CoverLetter Endpoints (routes/coverletter.py) ---
export const updateCoverLetterMetadata = (coverId, name) => {
     const params = new URLSearchParams({ 
        name: name
    });
    return apiClient.patch(`/coverletter/${coverId}/metadata?${params.toString()}`);
};
export const fetchCoverLetterDetails = (coverId) => apiClient.get(`/coverletter/${coverId}`);

// --- NEW AUTOFILL ENDPOINT ---
/**
 * Calls the non-destructive "Re-Classification Engine" on the backend.
 * @param {string} coverId - The ID of the cover letter
 * @param {string} strategy - "standard", "mission_driven", or "specialist"
 * @param {string} mode - "reset" (clears AI content) or "augment" (adds new)
 */
export const autofillCoverLetter = (coverId, strategy = 'standard', mode = 'reset') => {
    return apiClient.post(`/coverletter/${coverId}/autofill?strategy=${strategy}&mode=${mode}`);
};
// --- END NEW ENDPOINT ---

export const addCoverLetterIdea = (coverId, title, mappingPairIds, annotation) => { // <-- 1. Add annotation
    let params = `title=${encodeURIComponent(title)}`;
    
    (mappingPairIds || []).forEach(id => {
        params += `&mapping_pair_ids=${encodeURIComponent(id)}`;
    });

    if (annotation) { // <-- 2. Add annotation to params if it exists
        params += `&annotation=${encodeURIComponent(annotation)}`;
    }
    
    return apiClient.post(`/coverletter/${coverId}/idea?${params}`);
};


// --- UPDATED FUNCTION ---
export const addCoverLetterParagraph = (coverId, ideaIds, purpose, draftText = "", owner = "user", order = null) => {
    let params = `purpose=${encodeURIComponent(purpose)}`;
    
    if (ideaIds && ideaIds.length > 0) {
        ideaIds.forEach(id => {
            params += `&idea_ids=${encodeURIComponent(id)}`;
        });
    }
    
    if (draftText) params += `&draft_text=${encodeURIComponent(draftText)}`;
    if (owner) params += `&owner=${encodeURIComponent(owner)}`;
    if (order !== null) params += `&order=${order}`;

    return apiClient.post(`/coverletter/${coverId}/paragraph?${params}`);
};

// --- Prompt Endpoints (routes/prompt.py) ---
export const generateCvPrompt = (baseCvId, jobId, selectedSkillIds) => { // <-- 1. Add selectedSkillIds
    const params = new URLSearchParams({ base_cv_id: baseCvId, job_id: jobId });
    
    // 2. Append all skill IDs to the query params
    if (selectedSkillIds && selectedSkillIds.length > 0) {
        selectedSkillIds.forEach(id => {
            params.append('selected_skill_ids', id);
        });
    }
    
    // 3. Send the request
    return apiClient.post(`/prompt/generate-cv-prompt?${params.toString()}`);
};

export const generateCoverLetterPrompt = (mappingId) => {
    const params = new URLSearchParams({ mapping_id: mappingId });
    return apiClient.post(`/prompt/generate-coverletter-prompt?${params.toString()}`);
};

export const fetchAppSuiteData = () => apiClient.get('/application/app-suite-data/');

// --- ADD THIS FUNCTION ---
export const updateJob = (jobId, data) => {
    // data should be { title: "New Title", company: "New Co" }
    return apiClient.patch(`/job/${jobId}`, data);
};


// --- ADD THIS FUNCTION ---
export const deleteJobFeature = (jobId, featureId) => {
    return apiClient.delete(`/job/${jobId}/feature/${featureId}`);
};

// --- NEW UPSERT FUNCTION ---
export const upsertJob = (jobData) => {
    // jobData is the full JobUpsertPayload
    return apiClient.post('/job/upsert', jobData);
};


// --- ADD THIS FUNCTION ---
export const deleteJob = (jobId) => {
  return apiClient.delete(`/job/${jobId}`);
};

// --- ADD THIS FUNCTION ---
export const deleteApplication = (appId) => {
  return apiClient.delete(`/application/${appId}`);
};



// --- *** NEW FUNCTION TO CALL THE INFERER *** ---
/**
 * Calls the inference engine for a given mapping.
 * @param {string} mappingId - The ID of the mapping.
 *a * @param {string} mode - The tuning mode (e.g., "eager_mode", "picky_mode").
 */
export const inferMappingPairs = (mappingId, mode = "balanced_default") => {
    const params = new URLSearchParams({ mode });
    // This is a POST request, and the mode is a query parameter
    return apiClient.post(`/mapping/${mappingId}/infer?${params.toString()}`);
};
// --- *** END OF NEW FUNCTION *** ---


// --- NEW ---
export const updateCoverLetterIdea = (coverId, ideaId, ideaData) => {
    // ideaData is an object like { title, mapping_pair_ids }
    return apiClient.patch(`/coverletter/${coverId}/idea/${ideaId}`, ideaData);
};

// --- NEW ---
export const deleteCoverLetterIdea = (coverId, ideaId) => {
    return apiClient.delete(`/coverletter/${coverId}/idea/${ideaId}`);
};


// --- NEW ---
export const updateCoverLetterParagraph = (coverId, paraId, paraData) => {
    // paraData is an object like { purpose, idea_ids }
    return apiClient.patch(`/coverletter/${coverId}/paragraph/${paraId}`, paraData);
};

// --- NEW ---
export const deleteCoverLetterParagraph = (coverId, paraId) => {
    return apiClient.delete(`/coverletter/${coverId}/paragraph/${paraId}`);
};

// --- NEW: CONTEXT ASSEMBLER ENDPOINT ---
export const fetchCoverLetterPromptPayload = (coverId) => {
    return apiClient.get(`/prompt/coverletter-payload/${coverId}`);
};



// --- FORENSIC & ROLECASE ENDPOINTS ---

/**
 * 1. Generate RoleCase (Scratch): 
 * Runs Inference -> Saves Mapping -> Calculates Forensics.
 */
export const generateRoleCase = (jobId, cvId, mode = "balanced_default") => {
    return apiClient.post(`/forensics/generate`, { 
      job_id: jobId, 
      cv_id: cvId,
      mode 
    });
};

/**
 * 2. Get Existing Analysis (Read-Only):
 * fast fetch for returning users.
 */
export const fetchForensicAnalysis = (appId) => {
    // FIX: Added /forensics prefix
    return apiClient.get(`/forensics/applications/${appId}/forensic-analysis`);
};

/**
 * 3. Reject Match (Interactive):
 * Removes a match and returns the updated analysis instantly.
 */
export const rejectMatch = (appId, featureId) => {
    // FIX: Added /forensics prefix
    return apiClient.post(`/forensics/applications/${appId}/mappings/${featureId}/reject`);
};

/**
 * 4. Manual Match (Override):
 * User forces a specific text/item as evidence.
 * payload: { evidence_text: "...", cv_item_id: "optional", ... }
 */
export const createManualMatch = (appId, featureId, payload) => {
    // FIX: Added /forensics prefix
    return apiClient.post(`/forensics/applications/${appId}/mappings/${featureId}/manual`, payload);
};

export const promoteMatch = (appId, featureId, alternativeId) => {
    // FIX: Added /forensics prefix
    return apiClient.post(`/forensics/applications/${appId}/mappings/${featureId}/promote`, { alternative_id: alternativeId });
};

export const approveMatch = (appId, featureId) => {
    // FIX: Added /forensics prefix
    return apiClient.post(`/forensics/applications/${appId}/mappings/${featureId}/approve`);
};

export const fetchMatchPreview = async (jobId, cvId) => {
    // This calls the new lightweight endpoint
    const response = await apiClient.get(`/job/${jobId}/match-preview`, {
        params: { cv_id: cvId }
    });
    return response.data; // Expects { score: 90.0, badges: [...] }
};

export const getMatchPreview = (jobId, cvId) => apiClient.post(`/job/${jobId}/match-preview?cv_id=${cvId}`);

// [NEW] Update Feature (For Inline Editing)
// Note: If your backend doesn't support PATCH feature directly, 
// you might need to implement "delete then add" in the backend or frontend.
// Assuming backend has: PATCH /job/{job_id}/feature/{feature_id}
export const updateJobFeature = (jobId, featureId, data) => {
    // data = { description: "...", type: "..." }
    return apiClient.patch(`/job/${jobId}/feature/${featureId}`, data);
};


/**
 * Gets or creates a Derived CV for a specific application.
 * If force_refresh=true, it will regenerate the CV from the current mapping.
 */
export const getOrCreateDerivedCV = async (appId, options = {}) => {
    // options can include { force_refresh: true }
    const params = new URLSearchParams();
    if (options.force_refresh) params.append('force_refresh', 'true');
    
    const response = await apiClient.post(`/application/${appId}/tailored-cv?${params.toString()}`);
    return response.data;
};

/**
 * Updates a CV object directly.
 * This is used to save manual edits (summary, reordering) to a Derived CV.
 * Note: This might point to a generic CV update endpoint or a specific derived one.
 * Given your pattern, we'll map this to the generic CV update since DerivedCV inherits from CV.
 */
export const updateCV = async (cvId, cvData) => {
    // We reuse the generic CV update endpoint but pass the full object
    // You might need to ensure your backend PATCH /cv/{id} accepts complex nested data
    // OR create a specific endpoint for saving the tailored state.
    
    // For safety, let's use a specific endpoint for Derived CVs to avoid accidental data loss on Base CVs
    const response = await apiClient.put(`/cv/${cvId}/tailored-content`, cvData);
    return response.data;
};


/**
 * Trigger background scoring ONLY for jobs that haven't been scored yet.
 * This is efficient and non-blocking.
 */
export const scoreEmptyJobs = (cvId) => {
    // If cvId is provided, append it to the query string
    const params = cvId ? `?cv_id=${cvId}` : '';
    
    // Calls the new backend endpoint: POST /api/job/score-empty
    return apiClient.post(`/job/score-empty${params}`);
};

/**
 * Triggers the background inference/scoring for a specific Application.
 * Call this when the Dashboard loads and sees a score of 0 or missing analysis.
 * * Flow: 
 * 1. UI calls this (Fire & Forget).
 * 2. Backend enqueues task.
 * 3. Frontend waits for WebSocket 'APP_SCORED' event.
 */
export const triggerApplicationAnalysis = (appId) => {
    // Matches: backend/routes/forensics.py -> @router.post("/applications/{app_id}/generate-analysis")
    // Note: ensure the prefix matches where you mounted the forensics router. 
    // Usually mounted at /api/forensics
    return apiClient.post(`/forensics/applications/${appId}/generate-analysis`);
};