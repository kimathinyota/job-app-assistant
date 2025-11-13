// frontend/src/components/applications/SavedJobsView.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { 
    Plus, 
    Search, 
    Briefcase, 
    XCircle, 
    LayoutGrid, 
    Filter,
    CheckCircle2,
    CircleDashed
} from 'lucide-react';

import { 
    fetchAllJobs, 
    fetchAllApplications, 
    createMapping,
    createApplication,
    deleteJob
} from '../../api/applicationClient';
import { fetchAllCVs } from '../../api/cvClient';
import JobCard from './JobCard';
import JobModal from './JobModal'; 
import DeleteConfirmationModal from '../common/DeleteConfirmationModal'; 

const SavedJobsView = ({ defaultCvId, onNavigateToWorkspace }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [jobs, setJobs] = useState([]);
    const [applications, setApplications] = useState([]);
    const [cvs, setCvs] = useState([]);

    // --- UI States ---
    const [searchQuery, setSearchQuery] = useState('');
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    const [statusFilter, setStatusFilter] = useState('all'); // 'all', 'pending', 'started'

    // --- Modal States ---
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalJobId, setModalJobId] = useState(null); 
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [itemToDelete, setItemToDelete] = useState(null);

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

    // --- Memoized Data & Logic ---
    const applicationMap = useMemo(() => {
        const map = new Map();
        for (const app of applications) {
            map.set(app.job_id, app);
        }
        return map;
    }, [applications]);

    // 1. Compute Counts for Tabs
    const counts = useMemo(() => {
        const pending = jobs.filter(j => !applicationMap.has(j.id)).length;
        const started = jobs.filter(j => applicationMap.has(j.id)).length;
        return { all: jobs.length, pending, started };
    }, [jobs, applicationMap]);

    // 2. Filter & Sort Logic
    const processedJobs = useMemo(() => {
        let result = [...jobs];

        // A. Search Filter
        if (searchQuery.trim()) {
            const term = searchQuery.toLowerCase();
            result = result.filter(job => 
                job.title.toLowerCase().includes(term) || 
                job.company.toLowerCase().includes(term)
            );
        }

        // B. Status Filter
        if (statusFilter === 'pending') {
            result = result.filter(job => !applicationMap.has(job.id));
        } else if (statusFilter === 'started') {
            result = result.filter(job => applicationMap.has(job.id));
        }

        // C. Priority Sort (Pending First)
        // If 'all' is selected, we sort so that pending jobs are at the top
        if (statusFilter === 'all') {
            result.sort((a, b) => {
                const aHasApp = applicationMap.has(a.id);
                const bHasApp = applicationMap.has(b.id);
                if (aHasApp === bHasApp) return 0;
                return aHasApp ? 1 : -1; // Pending (false) comes before Started (true)
            });
        }

        return result;
    }, [jobs, searchQuery, statusFilter, applicationMap]);

    // --- Handlers ---
    const handleOpenAddModal = () => { setModalJobId(null); setIsModalOpen(true); };
    const handleOpenEditModal = (jobId) => { setModalJobId(jobId); setIsModalOpen(true); };
    const handleCloseModal = () => { setIsModalOpen(false); setModalJobId(null); };
    const handleJobUpdated = () => { loadData(); };
    const handleOpenDeleteModal = (job) => { setItemToDelete(job); setIsDeleteModalOpen(true); };
    const handleCloseDeleteModal = () => { setItemToDelete(null); setIsDeleteModalOpen(false); };

    const handleConfirmDelete = async () => {
        if (!itemToDelete) return;
        try {
            await deleteJob(itemToDelete.id);
            handleCloseDeleteModal();
            loadData(); 
        } catch (err) {
            alert(`Failed to delete job: ${err.message}`);
        }
    };

    const handleStartApplication = async (jobId, baseCvId) => {
        if (!baseCvId) return alert("Internal error: No CV ID provided.");
        try {
            const mappingRes = await createMapping(jobId, baseCvId);
            const appRes = await createApplication(jobId, baseCvId, mappingRes.data.id);
            onNavigateToWorkspace(appRes.data.id);
        } catch (err) {
            alert("Failed to start application.");
            console.error(err);
        }
    };

    if (loading) return <div className="text-center p-5 text-muted">Loading Library...</div>;
    if (error) return <p className="text-danger">{error}</p>;

    return (
        <div className="container-fluid px-0 pb-5">
             <style>
                {`
                .hover-lift { transition: transform 0.2s ease, box-shadow 0.2s ease; }
                .hover-lift:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0,0,0,0.08) !important; }
                .search-enter { animation: slideDown 0.2s ease-out; }
                @keyframes slideDown {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                /* Segmented Control Styles */
                .status-pill {
                    cursor: pointer;
                    font-size: 0.85rem;
                    font-weight: 500;
                    padding: 6px 12px;
                    border-radius: 20px;
                    transition: all 0.2s;
                    color: #64748b;
                    background: transparent;
                    border: 1px solid transparent;
                }
                .status-pill:hover {
                    background-color: #f1f5f9;
                    color: #0f172a;
                }
                .status-pill.active {
                    background-color: #0f172a; /* Brand Primary */
                    color: white;
                    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.2);
                }
                `}
            </style>

            {/* 1. Header Section */}
            <div className="d-flex flex-column flex-xl-row justify-content-between align-items-xl-center mb-4 gap-3">
                
                {/* Title & Stats */}
                <div className="d-flex align-items-center gap-3">
                    <div className="bg-primary bg-opacity-10 p-2 rounded-circle text-primary">
                        <LayoutGrid size={20} />
                    </div>
                    <div>
                        <h2 className="fw-bold text-dark mb-0 h5">Job Library</h2>
                        <p className="text-muted mb-0 small">
                            Manage your saved opportunities
                        </p>
                    </div>
                    
                    <div className="vr mx-2 d-none d-md-block opacity-25"></div>

                    {/* Modern Segmented Control / Filter Pills */}
                    <div className="d-flex gap-1 p-1 bg-white border rounded-pill shadow-sm">
                        <button 
                            className={`status-pill d-flex align-items-center gap-2 ${statusFilter === 'all' ? 'active' : ''}`}
                            onClick={() => setStatusFilter('all')}
                        >
                            All <span className="opacity-75 small">({counts.all})</span>
                        </button>
                        <button 
                            className={`status-pill d-flex align-items-center gap-2 ${statusFilter === 'pending' ? 'active' : ''}`}
                            onClick={() => setStatusFilter('pending')}
                        >
                            <CircleDashed size={14}/> To Apply <span className="opacity-75 small">({counts.pending})</span>
                        </button>
                        <button 
                            className={`status-pill d-flex align-items-center gap-2 ${statusFilter === 'started' ? 'active' : ''}`}
                            onClick={() => setStatusFilter('started')}
                        >
                            <CheckCircle2 size={14}/> Applied <span className="opacity-75 small">({counts.started})</span>
                        </button>
                    </div>
                </div>

                {/* Actions */}
                <div className="d-flex gap-2">
                    <button 
                        onClick={() => setIsSearchOpen(!isSearchOpen)}
                        className={`btn btn-white bg-white border shadow-sm d-flex align-items-center gap-2 text-muted hover-lift ${isSearchOpen || searchQuery ? 'text-primary border-primary' : ''}`}
                    >
                        <Search size={16}/> <span className="d-none d-sm-inline">Search</span>
                    </button>
                    <button 
                        className="btn btn-primary d-flex align-items-center gap-2 shadow-sm hover-lift" 
                        onClick={handleOpenAddModal}
                    >
                        <Plus size={18}/> Add Job
                    </button>
                </div>
            </div>

            {/* 2. Search Bar */}
            {isSearchOpen && (
                <div className="mb-4 search-enter">
                    <div className="input-group shadow-sm border rounded-3 overflow-hidden">
                        <span className="input-group-text bg-white border-0 ps-3">
                            <Filter size={18} className="text-muted"/>
                        </span>
                        <input 
                            type="text" 
                            className="form-control border-0 py-2 shadow-none" 
                            placeholder="Filter jobs by title or company..." 
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            autoFocus
                        />
                        {searchQuery && (
                            <button className="btn btn-white border-0 text-muted" onClick={() => setSearchQuery('')}>
                                <XCircle size={16}/>
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* 3. Job Grid */}
            <div className="row g-4">
                {processedJobs.length === 0 ? (
                    <div className="col-12">
                        <div className="text-center py-5 bg-light rounded-3 border border-dashed">
                            <div className="mb-3 text-muted opacity-50">
                                <Briefcase size={48} />
                            </div>
                            <h5 className="fw-bold text-muted">No jobs found</h5>
                            <p className="text-muted small">
                                {searchQuery ? "Try adjusting your search." : 
                                 statusFilter !== 'all' ? `No jobs matching '${statusFilter}' status.` :
                                 "Build your library by adding a job posting."}
                            </p>
                            {searchQuery || statusFilter !== 'all' ? (
                                <button className="btn btn-link text-primary" onClick={() => {setSearchQuery(''); setStatusFilter('all');}}>Clear Filters</button>
                            ) : (
                                <button className="btn btn-primary btn-sm mt-2" onClick={handleOpenAddModal}>+ Add Your First Job</button>
                            )}
                        </div>
                    </div>
                ) : (
                    processedJobs.map(job => {
                        const application = applicationMap.get(job.id);
                        return (
                            <div key={job.id} className="col-12 col-md-6 col-xl-4 d-flex align-items-stretch">
                                <div className="w-100 h-100 hover-lift transition-all">
                                    <JobCard
                                        job={job}
                                        cvs={cvs} 
                                        defaultCvId={defaultCvId} 
                                        application={application} 
                                        onStartApplication={handleStartApplication}
                                        onEdit={() => handleOpenEditModal(job.id)} 
                                        onDelete={() => handleOpenDeleteModal(job)}
                                    />
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* 4. Modals */}
            <JobModal
                key={modalJobId || 'new'} 
                initialJobId={modalJobId}
                isOpen={isModalOpen}
                onClose={handleCloseModal}
                onJobUpdated={handleJobUpdated}
            />

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