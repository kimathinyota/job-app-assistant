// frontend/src/components/applications/SavedJobsView.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { 
    fetchAllJobs, 
    fetchAllApplications, 
    createJob,
    createMapping,
    createApplication
} from '../../api/applicationClient';
import { fetchAllCVs } from '../../api/cvClient'; // <-- 1. IMPORT THE CV FETCHER
import JobCard from './JobCard';
import AddJobModal from './AddJobModal';
import JobEditorModal from './JobEditorModal'; 

const SavedJobsView = ({ defaultCvId, onNavigateToWorkspace }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [jobs, setJobs] = useState([]);
    const [applications, setApplications] = useState([]);
    const [isAddModalOpen, setIsAddModalOpen] = useState(false);
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [editingJobId, setEditingJobId] = useState(null);
    
    // 2. Add state to hold the list of CVs
    const [cvs, setCvs] = useState([]);

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            // 3. --- THIS IS THE FIX ---
            // Fetch all 3 data sources, including CVs
            const [cvsRes, appsRes, jobsRes] = await Promise.all([
                fetchAllCVs(), // <-- Added this call
                fetchAllApplications(),
                fetchAllJobs(),
                
            ]);
            
            setJobs(jobsRes.data || []);
            setApplications(appsRes.data || []);
            setCvs(cvsRes || []); // <-- 4. Save the list of CVs to state

        } catch (err) {
            setError("Failed to load data.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    // Create a lookup map to efficiently find the application for a job.
    const applicationMap = useMemo(() => {
        const map = new Map();
        for (const app of applications) {
            map.set(app.job_id, app);
        }
        return map;
    }, [applications]);

    const handleAddJob = async (title, company) => {
        if (!title || !company) return;
        try {
            await createJob(title, company);
            setIsAddModalOpen(false);
            loadData(); 
        } catch (err) {
            alert("Failed to create job.");
            console.error(err);
        }
    };
    
    const handleOpenEditor = (jobId) => {
        setEditingJobId(jobId);
        setIsEditModalOpen(true);
    };

    const handleCloseEditor = () => {
        setEditingJobId(null);
        setIsEditModalOpen(false);
    };
    
    const handleJobUpdated = () => {
        loadData(); 
    };

    const handleStartApplication = async (jobId, baseCvId) => {
        if (!baseCvId) { 
            alert("Internal error: No CV ID provided.");
            return;
        }
        
        try {
            const mappingRes = await createMapping(jobId, baseCvId);
            const mappingId = mappingRes.data.id;

            const appRes = await createApplication(jobId, baseCvId, mappingId);
            const applicationId = appRes.data.id;

            onNavigateToWorkspace(applicationId);
        } catch (err) {
            alert("Failed to start application.");
            console.error(err);
        }
    };

    if (loading) return <p>Loading saved jobs...</p>;
    if (error) return <p className="text-danger">{error}</p>;

    return (
        <div>
            <div className="d-flex justify-content-between align-items-center mb-3">
                <p className="text-muted">
                    Save and edit jobs here. Select a CV for each job to begin.
                </p>
                <button 
                    className="btn btn-primary" 
                    onClick={() => setIsAddModalOpen(true)}
                >
                    + Add New Job
                </button>
            </div>

            <div className="list-group">
                {jobs.map(job => {
                    // Find the specific application for this job
                    const application = applicationMap.get(job.id);
                    return (
                        <JobCard
                            key={job.id}
                            job={job}
                            cvs={cvs} // <-- 5. Pass the full list of CVs
                            defaultCvId={defaultCvId} 
                            application={application} // Pass the application object
                            onStartApplication={handleStartApplication}
                            onEdit={() => handleOpenEditor(job.id)}
                        />
                    );
                })}
            </div>
            
            <AddJobModal
                isOpen={isAddModalOpen}
                onClose={() => setIsAddModalOpen(false)}
                onSubmit={handleAddJob}
            />
            
            <JobEditorModal
                jobId={editingJobId}
                isOpen={isEditModalOpen}
                onClose={handleCloseEditor}
                onJobUpdated={handleJobUpdated}
            />
        </div>
    );
};

export default SavedJobsView;