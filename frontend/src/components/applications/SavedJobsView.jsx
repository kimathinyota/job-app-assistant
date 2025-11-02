// frontend/src/components/applications/SavedJobsView.jsx
import React, { useState, useEffect } from 'react';
import { 
    fetchAppSuiteData,
    createJob,
    createMapping,
    createApplication
} from '../../api/applicationClient';
import JobCard from './JobCard';
import AddJobModal from './AddJobModal';
import JobEditorModal from './JobEditorModal'; // --- 1. Import the new modal ---

const SavedJobsView = ({ defaultCvId, onNavigateToWorkspace }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [jobs, setJobs] = useState([]);
    const [applications, setApplications] = useState([]);
    const [isModalOpen, setIsModalOpen] = useState(false);

    // --- 2. Add state for BOTH modals ---
    const [isAddModalOpen, setIsAddModalOpen] = useState(false);
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [editingJobId, setEditingJobId] = useState(null);

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            // --- 2. REPLACE the Promise.all ---
            // OLD:
            // const [jobsRes, appsRes] = await Promise.all([
            //     fetchAllJobs(),
            //     fetchAllApplications()
            // ]);
            // setJobs(jobsRes.data || []);
            // setApplications(appsRes.data || []);

            // NEW:
            const res = await fetchAppSuiteData();
            setJobs(res.data.jobs || []);
            setApplications(res.data.applications || []);
            // --- End of change ---

        } catch (err) {
            setError("Failed to load app suite data."); // Updated error message
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    const handleAddJob = async (title, company) => {
        if (!title || !company) return;
        try {
            await createJob(title, company);
            setIsModalOpen(false);
            setIsAddModalOpen(false); // Use specific setter
            loadData(); // Refresh the list
        } catch (err) {
            alert("Failed to create job.");
            console.error(err);
        }
    };
    // --- 3. Add handlers for the new modal ---
    const handleOpenEditor = (jobId) => {
        setEditingJobId(jobId);
        setIsEditModalOpen(true);
    };

    const handleCloseEditor = () => {
        setEditingJobId(null);
        setIsEditModalOpen(false);
    };
    const handleJobUpdated = () => {
        // This is called by the modal after a successful update
        loadData(); // Just reload the main list
    };
    // --- End of new handlers ---

    const handleStartApplication = async (jobId) => {
        if (!defaultCvId) {
            alert("No default CV found. Please create a CV first in the CV Manager.");
            return;
        }
        try {
            // 1. Create the Mapping
            const mappingRes = await createMapping(jobId, defaultCvId);
            const mappingId = mappingRes.data.id;

            // 2. Create the Application
            const appRes = await createApplication(jobId, defaultCvId, mappingId);
            const applicationId = appRes.data.id;

            // 3. Navigate to the new workspace
            onNavigateToWorkspace(applicationId);
        } catch (err) {
            alert("Failed to start application.");
            console.error(err);
        }
    };

    const appliedJobIds = new Set(applications.map(app => app.job_id));

    if (loading) return <p>Loading saved jobs...</p>;
    if (error) return <p className="text-danger">{error}</p>;

    return (
        <div>
            <div className="d-flex justify-content-between align-items-center mb-3">
                <p className="text-muted">
                    Save and edit jobs here *before* starting an application.
                </p>
                <button 
                    className="btn btn-primary" 
                    onClick={() => setIsAddModalOpen(true)} // Use specific setter
                >
                    + Add New Job
                </button>
            </div>

            <div className="list-group">
                {jobs.map(job => (
                    <JobCard
                        key={job.id}
                        job={job}
                        hasApplication={appliedJobIds.has(job.id)}
                        onStartApplication={() => handleStartApplication(job.id)}
                        onEdit={() => handleOpenEditor(job.id)} // --- 4. Pass the handler
                    />
                ))}
            </div>
            
            <AddJobModal
                isOpen={isAddModalOpen} // Use specific state
                onClose={() => setIsAddModalOpen(false)} // Use specific setter
                onSubmit={handleAddJob}
            />
            
            {/* --- 5. Render the new modal --- */}
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