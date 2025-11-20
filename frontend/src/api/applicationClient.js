// frontend/src/api/applicationClient.js
import axios from 'axios';

const API_BASE_URL = 'http://192.168.1.161:8000/api'; // <-- THIS IS THE FIX

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});



// --- JobDescription Endpoints (routes/job.py) ---
export const fetchAllJobs = () => apiClient.get('/job/');
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

