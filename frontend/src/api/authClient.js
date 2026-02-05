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

// 1. LOGIN: This is NOT an axios call. 
// It returns the full URL for your <a> tag or window.location.href.
// This prevents the "404" error by dynamically building the correct path.
export const getGoogleLoginUrl = () => {
    return `${API_BASE_URL}/auth/login/google`;
};

// 2. CHECK SESSION: Standard Axios GET
// Used to see if the user is already logged in (calls /api/auth/me)
export const getCurrentUser = () => {
    return apiClient.get('/auth/me');
};

// 3. LOGOUT: Standard Axios POST
// Calls /api/auth/logout to clear the httpOnly cookie
export const logout = () => {
    return apiClient.post('/auth/logout');
};


