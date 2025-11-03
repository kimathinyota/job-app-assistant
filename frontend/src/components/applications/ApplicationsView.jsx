// frontend/src/components/applications/ApplicationsView.jsx
import React, { useState, useEffect } from 'react';
import {
  fetchAllApplications,
  fetchAllJobs,
  deleteApplication, // <-- We only need deleteApplication
} from '../../api/applicationClient';
import ApplicationCard from './ApplicationCard';
import DeleteConfirmationModal from '../common/DeleteConfirmationModal';

const ApplicationsView = ({ onNavigateToWorkspace }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState({ applications: [], jobs: [] });

  // --- Delete Modal State ---
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState(null); // Will be { application, job }

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch fresh data
      const [appsRes, jobsRes] = await Promise.all([
        fetchAllApplications(),
        fetchAllJobs(),
      ]);
      setData({
        applications: appsRes.data || [],
        jobs: jobsRes.data || [],
      });
    } catch (err) {
      setError('Failed to load applications.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const getJob = (jobId) => data.jobs.find((j) => j.id === jobId) || {};

  // --- Delete Modal Handlers ---
  const handleOpenDeleteModal = (application, job) => {
    setItemToDelete({ application, job });
    setIsDeleteModalOpen(true);
  };

  const handleCloseDeleteModal = () => {
    setItemToDelete(null);
    setIsDeleteModalOpen(false);
  };

  const handleConfirmDelete = async () => {
    if (!itemToDelete) return;
    try {
      // This ONLY deletes the application, not the job
      await deleteApplication(itemToDelete.application.id);
      handleCloseDeleteModal();
      loadData(); // Refresh the list
    } catch (err) {
      alert(`Failed to delete application: ${err.message}`);
      console.error(err);
    }
  };

  const columns = {
    draft: {
      title: 'Drafts',
      apps: data.applications.filter((a) => a.status === 'draft'),
    },
    applied: {
      title: 'Applied',
      apps: data.applications.filter((a) => a.status === 'applied'),
    },
    interview: {
      title: 'Interviewing',
      apps: data.applications.filter((a) => a.status === 'interview'),
    },
    offer: {
      title: 'Offers',
      apps: data.applications.filter((a) => a.status === 'offer'),
    },
    rejected: {
      title: 'Rejected',
      apps: data.applications.filter((a) => a.status === 'rejected'),
    },
  };

  if (loading) return <p>Loading applications...</p>;
  if (error) return <p className="text-danger">{error}</p>;

  return (
    <div className="row g-3">
      {Object.entries(columns).map(([key, col]) => (
        <div key={key} className="col-12 col-md-6 col-lg-4 col-xl">
          <div className="card bg-light h-100">
            <div className="card-header fw-bold text-capitalize">
              {col.title} ({col.apps.length})
            </div>
            <div className="card-body">
              {col.apps.length === 0 ? (
                <p className="small text-muted fst-italic">No applications</p>
              ) : (
                col.apps.map((app) => {
                  // --- THIS IS THE FIX ---
                  // Define 'job' here so it's in scope for the 'onDelete' handler
                  const job = getJob(app.job_id);
                  return (
                    <ApplicationCard
                      key={app.id}
                      application={app}
                      job={job}
                      onClick={() => onNavigateToWorkspace(app.id)}
                      onDelete={() => handleOpenDeleteModal(app, job)}
                    />
                  );
                })
              )}
            </div>
          </div>
        </div>
      ))}

      {/* Render the delete modal */}
      {itemToDelete && (
        <DeleteConfirmationModal
          isOpen={isDeleteModalOpen}
          onClose={handleCloseDeleteModal}
          onConfirm={handleConfirmDelete}
          itemName={itemToDelete.job.title} // Use job title for confirmation
          itemType="application"
        />
      )}
    </div>
  );
};

export default ApplicationsView;