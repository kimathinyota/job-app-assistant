import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Make sure this matches your FastAPI port

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// --- BASE CV CRUD ---

export const fetchAllCVs = async () => {
    const response = await apiClient.get('/cv/');
    return response.data;
};

// NEW: Fetches the full details of a single CV
export const fetchCVDetails = async (cvId) => {
    const response = await apiClient.get(`/cv/${cvId}`);
    return response.data; // Should return the full CV object with nested lists
};


export const createBaseCV = async (name, summary) => {
    const params = new URLSearchParams({ name, summary: summary || '' });
    const response = await apiClient.post(`/cv/?${params.toString()}`);
    return response.data;
};

/**
 * --- NEWLY ADDED ---
 * Updates the base details (name, summary) of a CV.
 * Maps to: PATCH /cv/{cvId}
 */
export const updateBaseCV = async (cvId, updateData) => {
    // updateData should be an object like { name: "New Name", summary: "New Summary" }
    const response = await apiClient.patch(`/cv/${cvId}`, updateData);
    return response.data;
};


export const deleteBaseCV = async (cvId) => {
    const response = await apiClient.delete(`/cv/${cvId}`);
    return response.data;
};


// --- NESTED ITEM ADDITION (CREATE) ---
// Note: Uses query parameters (params in Axios config)

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

// *** --- NEWLY ADDED --- ***
export const addHobby = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/hobby`, null, { params: data });

// --- NESTED ITEM DELETION ---

// Deletes a specific nested item (Experience, Skill, etc.)
export const deleteNestedItem = (cvId, itemId, listName) => 
    apiClient.delete(`/cv/${cvId}/${listName}/${itemId}`);


// --- NESTED LINKING / UNLINKING ---

// Links a master skill to a context (Experience, Project, Achievement)
export const linkSkill = (cvId, entityId, skillId, entityType) => {
    // entityType should be the plural form used in the URL (e.g., 'experiences', 'projects', 'achievements')
    return apiClient.post(`/cv/${cvId}/${entityType}/${entityId}/skill/${skillId}`);
};

// Links a master achievement to a context (Experience, Project)
export const linkAchievement = (cvId, entityId, achId, entityType) => {
    // entityType should be the plural form (e.g., 'experiences', 'projects')
    return apiClient.post(`/cv/${cvId}/${entityType}/${entityId}/achievement/${achId}`);
};

// --- Add more client methods for nested UPDATE (PATCH) and UNLINK (DELETE) here ---

export default apiClient; // Ensure apiClient is exported if needed elsewhere, though usually named exports are preferred.