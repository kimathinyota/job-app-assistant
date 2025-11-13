// frontend/src/components/applications/ApplicationsView.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { 
    Plus, 
    Filter, 
    Briefcase, 
    CheckCircle, 
    XCircle, 
    Clock, 
    FileText, 
    Layout,
    Search,
    Eye,
    EyeOff
} from 'lucide-react';

import {
  fetchAllApplications,
  fetchAllJobs,
  deleteApplication,
} from '../../api/applicationClient';
import ApplicationCard from './ApplicationCard';
import DeleteConfirmationModal from '../common/DeleteConfirmationModal';

// 1. Accept onTrackJob prop
const ApplicationsView = ({ onNavigateToWorkspace, onTrackJob }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState({ applications: [], jobs: [] });

  // --- Smart View States ---
  const [showEmptyStages, setShowEmptyStages] = useState(false); // Default to hidden
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // --- Delete Modal State ---
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState(null); 

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
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

  // --- Handlers ---
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
      await deleteApplication(itemToDelete.application.id);
      handleCloseDeleteModal();
      loadData(); 
    } catch (err) {
      alert(`Failed to delete application: ${err.message}`);
    }
  };

  // --- Smart Logic: Filter & Sort ---
  const filteredApps = useMemo(() => {
    if (!searchQuery.trim()) return data.applications;
    
    return data.applications.filter(app => {
        const job = getJob(app.job_id);
        const term = searchQuery.toLowerCase();
        return (
            (job.title && job.title.toLowerCase().includes(term)) ||
            (job.company && job.company.toLowerCase().includes(term))
        );
    });
  }, [data.applications, searchQuery, data.jobs]);

  // --- Column Configuration ---
  const allColumns = [
    {
        id: 'draft',
        title: 'Drafts',
        icon: FileText,
        color: 'text-secondary',
        bgColor: 'bg-secondary',
        statusMatch: 'draft'
    },
    {
        id: 'applied',
        title: 'Applied',
        icon: Briefcase,
        color: 'text-primary',
        bgColor: 'bg-primary',
        statusMatch: 'applied'
    },
    {
        id: 'interview',
        title: 'Interviewing',
        icon: Clock,
        color: 'text-info',
        bgColor: 'bg-info',
        statusMatch: 'interview'
    },
    {
        id: 'offer',
        title: 'Offers',
        icon: CheckCircle,
        color: 'text-success',
        bgColor: 'bg-success',
        statusMatch: 'offer'
    },
    {
        id: 'rejected',
        title: 'Rejected',
        icon: XCircle,
        color: 'text-danger',
        bgColor: 'bg-danger',
        statusMatch: 'rejected'
    },
  ];

  // 1. Map apps to columns
  const populatedColumns = allColumns.map(col => ({
      ...col,
      apps: filteredApps.filter(a => a.status === col.statusMatch)
  }));

  // 2. Determine which columns to actually show
  const activeColumns = showEmptyStages 
      ? populatedColumns 
      : populatedColumns.filter(col => col.apps.length > 0);

  // 3. Count how many are hidden (for user feedback)
  const hiddenCount = populatedColumns.length - activeColumns.length;

  if (loading) return <div className="text-center p-5 text-muted">Loading Pipeline...</div>;
  if (error) return <p className="text-danger">{error}</p>;

  return (
    <div className="container-fluid px-0 h-100">
        <style>
            {`
            .hover-lift {
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .hover-lift:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.08) !important;
            }
            .cursor-pointer { cursor: pointer; }
            .search-enter {
                animation: slideDown 0.2s ease-out;
            }
            @keyframes slideDown {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            `}
        </style>

        {/* 1. Header Section */}
        <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-4 gap-3">
            <div>
                <h2 className="fw-bold text-dark mb-1 d-flex align-items-center gap-2">
                    <Layout className="text-primary" /> Application Pipeline
                </h2>
                <div className="d-flex align-items-center gap-2">
                    <p className="text-muted mb-0 small">
                        Total: <span className="fw-bold text-dark">{filteredApps.length}</span> applications.
                    </p>
                    {/* Smart Feedback: Tell user if we are hiding things */}
                    {!showEmptyStages && hiddenCount > 0 && (
                        <span className="badge bg-light text-muted border fw-normal cursor-pointer" onClick={() => setShowEmptyStages(true)}>
                            +{hiddenCount} empty stages hidden
                        </span>
                    )}
                </div>
            </div>

            <div className="d-flex gap-2">
                {/* Toggle Empty Stages */}
                <button 
                    onClick={() => setShowEmptyStages(!showEmptyStages)}
                    className={`btn btn-white bg-white border shadow-sm d-flex align-items-center gap-2 text-muted hover-lift ${showEmptyStages ? 'text-primary border-primary' : ''}`}
                    title={showEmptyStages ? "Hide empty columns" : "Show all columns"}
                >
                    {showEmptyStages ? <EyeOff size={16}/> : <Eye size={16}/>}
                    <span className="d-none d-sm-inline">{showEmptyStages ? 'Hide Empty' : 'Show All'}</span>
                </button>

                {/* Filter Toggle */}
                <button 
                    onClick={() => setIsFilterOpen(!isFilterOpen)}
                    className={`btn btn-white bg-white border shadow-sm d-flex align-items-center gap-2 text-muted hover-lift ${isFilterOpen || searchQuery ? 'text-primary border-primary' : ''}`}
                >
                    <Filter size={16}/> 
                    <span className="d-none d-sm-inline">Filter</span>
                </button>

                {/* 2. Action Button: Navigates to Saved Jobs */}
                <button 
                    onClick={onTrackJob} 
                    className="btn btn-primary d-flex align-items-center gap-2 shadow-sm hover-lift"
                >
                    <Plus size={18}/> Track Job
                </button>
            </div>
        </div>

        {/* 2. Search Bar (Conditional Render) */}
        {isFilterOpen && (
            <div className="mb-4 search-enter">
                <div className="input-group shadow-sm border rounded-3 overflow-hidden">
                    <span className="input-group-text bg-white border-0 ps-3">
                        <Search size={18} className="text-muted"/>
                    </span>
                    <input 
                        type="text" 
                        className="form-control border-0 py-2 shadow-none" 
                        placeholder="Filter by company or job title..." 
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        autoFocus
                    />
                    {searchQuery && (
                        <button 
                            className="btn btn-white border-0 text-muted"
                            onClick={() => setSearchQuery('')}
                        >
                            <XCircle size={16}/>
                        </button>
                    )}
                </div>
            </div>
        )}

        {/* 3. Dynamic Grid */}
        <div className="row g-4">
            {activeColumns.length === 0 ? (
                // Empty State (if everything is hidden or no data)
                <div className="col-12">
                    <div className="text-center py-5 bg-light rounded-3 border border-dashed">
                        <div className="mb-3 text-muted opacity-50">
                            <Briefcase size={48} />
                        </div>
                        <h5 className="fw-bold text-muted">No applications found</h5>
                        <p className="text-muted small">
                            {searchQuery ? "Try adjusting your search filters." : "Your pipeline is empty."}
                        </p>
                        {/* 3. Smart Action: Clear Search OR Track Job */}
                        {searchQuery ? (
                            <button className="btn btn-link text-primary" onClick={() => setSearchQuery('')}>Clear Filters</button>
                        ) : (
                            <button className="btn btn-primary btn-sm mt-2" onClick={onTrackJob}>Track your first job</button>
                        )}
                    </div>
                </div>
            ) : (
                activeColumns.map((col) => (
                    <div key={col.id} className="col-12 col-md-6 col-lg-4 col-xl">
                        <div className="h-100 d-flex flex-column animate-fade-in">
                            {/* Column Header */}
                            <div className="d-flex align-items-center justify-content-between mb-3 pb-2 border-bottom">
                                <div className="d-flex align-items-center gap-2">
                                    <col.icon size={18} className={col.color} />
                                    <h6 className="fw-bold text-dark mb-0 text-uppercase small">{col.title}</h6>
                                </div>
                                <span className={`badge rounded-pill ${col.bgColor} bg-opacity-10 ${col.color}`}>
                                    {col.apps.length}
                                </span>
                            </div>

                            {/* Column Content */}
                            <div className="flex-grow-1">
                                {col.apps.map((app) => {
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
                                })}
                            </div>
                        </div>
                    </div>
                ))
            )}
        </div>

      {/* Delete Confirmation Modal */}
      {itemToDelete && (
        <DeleteConfirmationModal
          isOpen={isDeleteModalOpen}
          onClose={handleCloseDeleteModal}
          onConfirm={handleConfirmDelete}
          itemName={itemToDelete.job.title} 
          itemType="application"
        />
      )}
    </div>
  );
};

export default ApplicationsView;