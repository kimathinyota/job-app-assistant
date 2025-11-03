// frontend/src/components/applications/SavedJobsView.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { 
    fetchAllJobs, 
    fetchAllApplications, 
    createMapping,
    createApplication,
    deleteJob
} from '../../api/applicationClient';
import { fetchAllCVs } from '../../api/cvClient';
import JobCard from './JobCard';
import JobModal from './JobModal'; // <-- 1. Import the new unified modal
import DeleteConfirmationModal from '../common/DeleteConfirmationModal'; // <-- 2. Import modal

// Delete imports for AddJobModal and JobEditorModal
// import AddJobModal from './AddJobModal';
// import JobEditorModal from './JobEditorModal'; 

const SavedJobsView = ({ defaultCvId, onNavigateToWorkspace }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [jobs, setJobs] = useState([]);
    const [applications, setApplications] = useState([]);
    const [cvs, setCvs] = useState([]);

    // --- 2. Simplified state for the modal ---
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalJobId, setModalJobId] = useState(null); // null = "Add" mode, 'job_id' = "Edit" mode

    // --- 3. Add state for delete modal ---
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [itemToDelete, setItemToDelete] = useState(null);

    // --- 4. Add handlers for delete modal ---
    const handleOpenDeleteModal = (job) => {
        setItemToDelete(job);
        setIsDeleteModalOpen(true);
    };

    const handleCloseDeleteModal = () => {
        setItemToDelete(null);
        setIsDeleteModalOpen(false);
    };

    const handleConfirmDelete = async () => {
        if (!itemToDelete) return;
        try {
        await deleteJob(itemToDelete.id);
        handleCloseDeleteModal();
        loadData(); // Refresh the list
        } catch (err) {
        alert(`Failed to delete job: ${err.message}`);
        console.error(err);
        }
    };

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [cvsRes, appsRes, jobsRes] = await Promise.all([
                fetchAllCVs(), 
                fetchAllApplications(),
                fetchAllJobs(),
            ]);
            
            setJobs(jobsRes.data || []);
            setApplications(appsRes.data || []);
            setCvs(cvsRes || []); 

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

    const applicationMap = useMemo(() => {
        const map = new Map();
        for (const app of applications) {
            map.set(app.job_id, app);
        }
        return map;
    }, [applications]);

    // --- 3. New handlers for the unified modal ---
    const handleOpenAddModal = () => {
        setModalJobId(null); // Set ID to null for "Add" mode
        setIsModalOpen(true);
    };

    const handleOpenEditModal = (jobId) => {
        setModalJobId(jobId); // Set the job ID for "Edit" mode
        setIsModalOpen(true);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setModalJobId(null);
    };
    
    const handleJobUpdated = () => {
        loadData(); // Just reload the main list
    };
    // --- End of new handlers ---

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
                    onClick={handleOpenAddModal} // <-- 4. Use new handler
                >
                    + Add New Job
                </button>
            </div>

            <div className="list-group">
                {jobs.map(job => {
                    const application = applicationMap.get(job.id);
                    return (
                        <JobCard
                            key={job.id}
                            job={job}
                            cvs={cvs} 
                            defaultCvId={defaultCvId} 
                            application={application} 
                            onStartApplication={handleStartApplication}
                            onEdit={() => handleOpenEditModal(job.id)} // <-- 5. Use new handler
                            onDelete={() => handleOpenDeleteModal(job)}
                        />
                    );
                })}
            </div>


            
            {/* --- 6. Render the new unified modal --- */}
            {/* We add 'key' to force React to re-create the component */}
            {/* when the modalJobId changes, ensuring state resets */}
            <JobModal
                key={modalJobId || 'new'} 
                initialJobId={modalJobId}
                isOpen={isModalOpen}
                onClose={handleCloseModal}
                onJobUpdated={handleJobUpdated}

            />

            {/* ... (existing JobModal) */}

            {/* --- 6. Render the delete modal --- */}
            {itemToDelete && (
                <DeleteConfirmationModal
                isOpen={isDeleteModalOpen}
                onClose={handleCloseDeleteModal}
                onConfirm={handleConfirmDelete}
                itemName={itemToDelete.title}
                itemType="job"
                />
            )}
        </div>
    );
};

export default SavedJobsView;