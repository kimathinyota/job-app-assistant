import axios from 'axios';

// The base URL MUST match the host/port defined in your backend/main.py
// const API_BASE_URL = 'http://192.168.1.161:8000/api'; // <-- THIS IS THE FIX

const API_BASE_URL = "http://localhost:8000/api"

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
    headers: {
        // We generally use application/json, but the POST /cv endpoint needs special handling (see createBaseCV)
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

/**
 * Imports a CV from raw text.
 * Maps to: POST /cv/import
 * Payload: { text: "Raw CV Text...", name: "Internal Label" }
 */
export const importCV = async (name, text) => {
    const response = await apiClient.post('/cv/import', { 
        name: name, 
        text: text 
    });
    return response.data;
};


export const createBaseCV = async (name, firstName, lastName, title, summary) => {
    const params = new URLSearchParams({ name });
    if (firstName) params.append('first_name', firstName);
    if (lastName) params.append('last_name', lastName);
    if (title) params.append('title', title);
    if (summary) params.append('summary', summary);
    
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


// --- *** NEW: Complex Experience Endpoints *** ---

/**
 * Creates a new Experience and all its dependencies (skills, achievements)
 * by sending the full complex payload to the backend.
 * @param {string} cvId - The ID of the CV.
 * @param {object} data - The ExperienceComplexPayload object.
 * @returns {Promise<object>} - The created Experience object.
 */
export const addExperienceComplex = (cvId, data) => {
    // We send 'data' as the request body, not as params
    return apiClient.post(`/cv/${cvId}/experience/complex`, data);
};

/**
 * Updates an existing Experience and all its dependencies.
 * @param {string} cvId - The ID of the CV.
 * @param {string} expId - The ID of the Experience to update.
 * @param {object} data - The ExperienceComplexPayload object.
 * @returns {Promise<object>} - The updated Experience object.
 */
export const updateExperienceComplex = (cvId, expId, data) => {
    // We send 'data' as the request body
    return apiClient.patch(`/cv/${cvId}/experience/${expId}/complex`, data);
};


// --- *** NEW: Complex Education Endpoints *** ---

/**
 * Creates a new Education item and all its dependencies (skills, achievements)
 * by sending the full complex payload to the backend.
 * @param {string} cvId - The ID of the CV.
 * @param {object} data - The EducationComplexPayload object.
 * @returns {Promise<object>} - The created Education object.
 */
export const addEducationComplex = (cvId, data) => {
    return apiClient.post(`/cv/${cvId}/education/complex`, data);
};

/**
 * Updates an existing Education item and all its dependencies.
 * @param {string} cvId - The ID of the CV.
 * @param {string} eduId - The ID of the Education item to update.
 * @param {object} data - The EducationComplexPayload object.
 * @returns {Promise<object>} - The updated Education object.
 */
export const updateEducationComplex = (cvId, eduId, data) => {
    return apiClient.patch(`/cv/${cvId}/education/${eduId}/complex`, data);
};

// --- *** NEW: Complex Hobby Endpoints *** ---

/**
 * Creates a new Hobby item and all its dependencies (skills, achievements)
 * by sending the full complex payload to the backend.
 * @param {string} cvId - The ID of the CV.
 * @param {object} data - The HobbyComplexPayload object.
 * @returns {Promise<object>} - The created Hobby object.
 */
export const addHobbyComplex = (cvId, data) => {
    return apiClient.post(`/cv/${cvId}/hobby/complex`, data);
};

/**
 * Updates an existing Hobby item and all its dependencies.
 * @param {string} cvId - The ID of the CV.
 * @param {string} hobbyId - The ID of the Hobby item to update.
 * @param {object} data - The HobbyComplexPayload object.
 * @returns {Promise<object>} - The updated Hobby object.
 */
export const updateHobbyComplex = (cvId, hobbyId, data) => {
    return apiClient.patch(`/cv/${cvId}/hobby/${hobbyId}/complex`, data);
};

// --- *** NEW: Complex Project Endpoints *** ---

/**
 * Creates a new Project item and all its dependencies (skills, achievements)
 * by sending the full complex payload to the backend.
 * @param {string} cvId - The ID of the CV.
 * @param {object} data - The ProjectComplexPayload object.
 * @returns {Promise<object>} - The created Project object.
 */
export const addProjectComplex = (cvId, data) => {
    return apiClient.post(`/cv/${cvId}/project/complex`, data);
};

/**
 * Updates an existing Project item and all its dependencies.
 * @param {string} cvId - The ID of the CV.
 * @param {string} projectId - The ID of the Project item to update.
 * @param {object} data - The ProjectComplexPayload object.
 * @returns {Promise<object>} - The updated Project object.
 */
export const updateProjectComplex = (cvId, projectId, data) => {
    return apiClient.patch(`/cv/${cvId}/project/${projectId}/complex`, data);
};
// --- NESTED ITEM ADDITION (CREATE) ---
// Note: Uses query parameters (params in Axios config)

// This simple version is now obsolete for CVManagerPage,
// but might be used elsewhere.
export const addEducation = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/education`, null, { params: data });

export const addSkill = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/skill`, null, { params: data });

export const addProject = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/project`, null, { params: data });

export const addAchievement = (cvId, data) => 
    apiClient.post(`/cv/${cvId}/achievement`, null, { params: data });

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

/**
 * Fetches all skills for an entity PLUS its children's skills.
 * Maps to: GET /cv/{cvId}/{listName}/{itemId}/skills/aggregated
 */
export const fetchAggregatedSkills = async (cvId, listName, itemId) => {
    const response = await apiClient.get(`/cv/${cvId}/${listName}/${itemId}/skills/aggregated`);
    return response.data; // Returns a list of Skill objects
};


// --- Add more client methods for nested UPDATE (PATCH) and UNLINK (DELETE) here ---

export default apiClient; // Ensure apiClient is exported if needed elsewhere, though usually named exports are preferred.