// frontend/src/api/goalClient.js
import axios from 'axios';

const API_BASE_URL = 'http://192.168.1.161:8000/api';
// const API_BASE_URL = "http://localhost:8000/api"

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// --- Goal Endpoints (routes/goal.py) ---

export const fetchAllGoals = () => apiClient.get('/goal/');

export const fetchGoalDetails = (goalId) => apiClient.get(`/goal/${goalId}`);

export const createGoal = (title, description, metric) => {
    const params = new URLSearchParams({ title });
    if (description) params.append('description', description);
    if (metric) params.append('metric', metric);
    return apiClient.post(`/goal/?${params.toString()}`);
};

export const updateGoal = (goalId, data) => apiClient.patch(`/goal/${goalId}`, data);

export const deleteGoal = (goalId) => apiClient.delete(`/goal/${goalId}`);

export const addWorkToGoal = (goalId, workId) => apiClient.post(`/goal/${goalId}/add-work/${workId}`);


// --- WorkItem Endpoints (routes/workitem.py) ---

export const fetchAllWorkItems = () => apiClient.get('/workitem/');

export const fetchWorkItemDetails = (workId) => apiClient.get(`/workitem/${workId}`);

export const createWorkItem = (title, type = 'research', relatedIds = {}) => {
    // Note: Backend expects 'work_type', not 'type'
    const params = new URLSearchParams({ title, work_type: type });
    
    // Add optional related IDs
    if (relatedIds.applicationId) params.append('related_application_id', relatedIds.applicationId);
    if (relatedIds.interviewId) params.append('related_interview_id', relatedIds.interviewId);
    if (relatedIds.jobId) params.append('related_job_id', relatedIds.jobId);
    if (relatedIds.goalId) params.append('related_goal_id', relatedIds.goalId);
    
    return apiClient.post(`/workitem/?${params.toString()}`);
};

export const updateWorkItem = (workId, data) => apiClient.patch(`/workitem/${workId}`, data);

export const deleteWorkItem = (workId) => apiClient.delete(`/workitem/${workId}`);

export const markWorkItemComplete = (workId, reflection) => {
    let url = `/workitem/${workId}/complete`;
    if (reflection) {
        url += `?reflection=${encodeURIComponent(reflection)}`;
    }
    return apiClient.post(url);
};