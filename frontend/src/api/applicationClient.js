// frontend/src/api/applicationClient.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Make sure this matches your FastAPI port

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

// --- Mapping Endpoints (routes/mapping.py) ---
export const fetchMappingDetails = (mappingId) => apiClient.get(`/mapping/${mappingId}`);
export const createMapping = (jobId, baseCvId) => {
    const params = new URLSearchParams({ job_id: jobId, base_cv_id: baseCvId });
    return apiClient.post(`/mapping/?${params.toString()}`);
};
export const addMappingPair = (mappingId, featureId, experienceId) => {
    const params = new URLSearchParams({ 
        feature_id: featureId, 
        experience_id: experienceId 
    });
    return apiClient.post(`/mapping/${mappingId}/pair?${params.toString()}`);
};
export const deleteMappingPair = (mappingId, pairId) => {
    // This route isn't in your files, but is implied by your registry
    // Assuming it's DELETE /mapping/{mapping_id}/pair/{pair_id}
    // If you don't have this, you'll need to add it to backend/routes/mapping.py
    // For now, we'll assume it exists.
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
export const addCoverLetterIdea = (coverId, title, mappingPairIds) => {
    // Note: FastAPI/Pydantic needs list items sent this way with query params
    let params = `title=${encodeURIComponent(title)}`;
    mappingPairIds.forEach(id => {
        params += `&mapping_pair_ids=${encodeURIComponent(id)}`;
    });
    return apiClient.post(`/coverletter/${coverId}/idea?${params}`);
};
export const addCoverLetterParagraph = (coverId, ideaIds, purpose) => {
    let params = `purpose=${encodeURIComponent(purpose)}`;
    ideaIds.forEach(id => {
        params += `&idea_ids=${encodeURIComponent(id)}`;
    });
    return apiClient.post(`/coverletter/${coverId}/paragraph?${params}`);
};


// --- Prompt Endpoints (routes/prompt.py) ---
export const generateCvPrompt = (baseCvId, jobId) => {
    const params = new URLSearchParams({ base_cv_id: baseCvId, job_id: jobId });
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