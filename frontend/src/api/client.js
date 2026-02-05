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

export default apiClient;


/**
 * Fetches a list of all existing CVs.
 * Maps to: GET /cv/
 */
export const fetchAllCVs = async () => {
    const response = await apiClient.get('/cv/');
    return response.data;
};

/**
 * Creates a new base CV record.
 * Maps to: POST /cv/?name=...&summary=...
 * Note: Your FastAPI route accepts name/summary as query parameters, not a JSON body.
 */
export const createBaseCV = async (name, summary) => {
    // URLSearchParams correctly handles encoding for query parameters
    const params = new URLSearchParams({
        name: name,
        summary: summary || '', // Ensure summary is handled even if optional
    });
    
    // Axios posts to the URL, using the parameters in the query string
    const response = await apiClient.post(`/cv/?${params.toString()}`);
    return response.data;
};