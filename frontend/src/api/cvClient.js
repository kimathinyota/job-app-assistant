// frontend/src/api/cvClient.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// --- BASE CV CRUD (from previous steps, included here for context) ---

export const fetchAllCVs = async () => {
    const response = await apiClient.get('/cv/');
    return response.data;
};

export const createBaseCV = async (name, summary) => {
    const params = new URLSearchParams({ name, summary: summary || '' });
    const response = await apiClient.post(`/cv/?${params.toString()}`);
    return response.data;
};

export const deleteBaseCV = async (cvId) => {
    const response = await apiClient.delete(`/cv/${cvId}`);
    return response.data;
};


// --- NESTED ITEM ADDITION (CREATE) ---
// Note: These use query parameters, matching the structure of your FastAPI endpoints

export const addExperience = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/experience`, null, { params: data });

export const addEducation = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/education`, null, { params: data });

export const addSkill = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/skill`, null, { params: data });

export const addProject = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/project`, null, { params: data });

export const addAchievement = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/achievement`, null, { params: data });


// --- LINKING / UNLINKING ---

// Links a master skill to a context (Experience, Project, Achievement)
export const linkSkill = (cvId, entityId, skillId, entityType) => {
    // entityType should be the plural form used in the URL (e.g., 'experiences', 'projects')
    return apiClient.post(`/cv/${cvId}/${entityType}/${entityId}/skill/${skillId}`);
};

// Deletes a nested item (Experience, Education, etc.)
export const deleteNestedItem = (cvId, itemId, listName) => 
    apiClient.delete(`/cv/${cvId}/${listName}/${itemId}`);

// --- Add more client methods here as needed for other nested CRUD/Linking operations ---

export default apiClient;
