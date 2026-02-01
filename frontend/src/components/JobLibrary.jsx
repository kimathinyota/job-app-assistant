import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom'; 
import { 
    Plus, 
    Search, 
    Briefcase, 
    XCircle, 
    LayoutGrid, 
    Filter,
    FileText,
    CheckCircle2,
    CircleDashed
} from 'lucide-react';

import { 
    fetchAllJobs, 
    fetchAllApplications, 
    createMapping,
    createApplication,
    deleteJob
} from '../api/applicationClient';
import { fetchAllCVs } from '../api/cvClient';

// Components
// frontend/src/components/applications/JobCard.jsx
import JobCard from './applications/JobCard';
import JobModal from './applications/JobModal'; 
import DeleteConfirmationModal from './common/DeleteConfirmationModal'; 

const JobLibrary = () => {
    const navigate = useNavigate();

    // --- Data State ---
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [jobs, setJobs] = useState([]);
    const [applications, setApplications] = useState([]);
    const [cvs, setCvs] = useState([]);

    // --- UI States ---
    const [searchQuery, setSearchQuery] = useState('');
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    
    // RESTORED: Status Filter State
    const [statusFilter, setStatusFilter] = useState('all'); // 'all' | 'pending' | 'started'

    // --- Modal States ---
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalJobId, setModalJobId] = useState(null); 
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [itemToDelete, setItemToDelete] = useState(null);
    const [viewingDescJob, setViewingDescJob] = useState(null);

    // --- 1. Load Data ---
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
            setError("Failed to load library.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    // --- 2. Computed Data ---
    
    const defaultCvId = useMemo(() => {
        if (cvs && cvs.length > 0) return cvs[0].id;
        return null;
    }, [cvs]);

    const applicationMap = useMemo(() => {
        const map = new Map();
        for (const app of applications) {
            map.set(app.job_id, app);
        }
        return map;
    }, [applications]);

    // RESTORED: Counts Logic
    const counts = useMemo(() => {
        const pending = jobs.filter(j => !applicationMap.has(j.id)).length;
        const started = jobs.filter(j => applicationMap.has(j.id)).length;
        return { all: jobs.length, pending, started };
    }, [jobs, applicationMap]);

    const processedJobs = useMemo(() => {
        let result = [...jobs];

        // 1. Text Search
        if (searchQuery.trim()) {
            const term = searchQuery.toLowerCase();
            result = result.filter(job => 
                job.title.toLowerCase().includes(term) || 
                job.company.toLowerCase().includes(term)
            );
        }

        // 2. RESTORED: Status Filtering
        if (statusFilter === 'pending') {
            result = result.filter(job => !applicationMap.has(job.id));
        } else if (statusFilter === 'started') {
            result = result.filter(job => applicationMap.has(job.id));
        }

        // 3. Sorting (Newest first, but prioritize applied if 'all' selected)
        result.sort((a, b) => {
            // If viewing 'all', grouping applied ones together can be nice, 
            // but strict date sorting is usually better for a library. 
            // Let's stick to simple reverse order (newest first).
            return b.id.localeCompare(a.id); 
        });

        return result; 
    }, [jobs, searchQuery, statusFilter, applicationMap]);

    // --- 3. Handlers ---

    const handleOpenAddModal = () => { setModalJobId(null); setIsModalOpen(true); };
    const handleOpenEditModal = (jobId) => { setModalJobId(jobId); setIsModalOpen(true); };
    const handleCloseModal = () => { setIsModalOpen(false); setModalJobId(null); };
    const handleJobUpdated = () => { loadData(); };
    
    const handleOpenDeleteModal = (job) => { setItemToDelete(job); setIsDeleteModalOpen(true); };
    const handleCloseDeleteModal = () => { setItemToDelete(null); setIsDeleteModalOpen(false); };

    const handleViewDescription = (job) => { setViewingDescJob(job); };
    const handleCloseDescModal = () => { setViewingDescJob(null); };

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

    const handleViewApplication = (applicationId) => {
        navigate(`/application/${applicationId}`);
    };

    const handleStartApplication = async (jobId, baseCvId) => {
        if (!baseCvId) return alert("Please select a CV to start.");
        try {
            const mappingRes = await createMapping(jobId, baseCvId);
            const appRes = await createApplication(jobId, baseCvId, mappingRes.data.id);
            navigate(`/application/${appRes.data.id}`);
        } catch (err) {
            alert("Failed to start application.");
            console.error(err);
        }
    };

    if (loading) return <div className="text-center p-5 text-muted">Loading Library...</div>;
    if (error) return <p className="text-danger p-4">{error}</p>;

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
                    background-color: #0f172a;
                    color: white;
                    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.2);
                }
                .job-desc-content h1, .job-desc-content h2, .job-desc-content h3 { font-size: 1.1rem; font-weight: bold; margin-top: 1rem; margin-bottom: 0.5rem; }
                .job-desc-content ul { padding-left: 1.5rem; margin-bottom: 1rem; }
                .job-desc-content p { margin-bottom: 0.75rem; }
                `}
            </style>

            {/* Header Section */}
            <div className="d-flex flex-column flex-xl-row justify-content-between align-items-xl-center mb-4 gap-3">
                <div className="d-flex align-items-center gap-3">
                    <div className="bg-primary bg-opacity-10 p-2 rounded-circle text-primary">
                        <Briefcase size={20} />
                    </div>
                    <div>
                        <h2 className="fw-bold text-dark mb-0 h5">Job Library</h2>
                        <p className="text-muted mb-0 small">
                            Manage your saved opportunities
                        </p>
                    </div>
                    <div className="vr mx-2 d-none d-md-block opacity-25"></div>
                    
                    {/* RESTORED: Toggle Pills */}
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

            {/* Search Bar */}
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

            {/* Job Grid */}
            <div className="row g-4">
                {processedJobs.length === 0 ? (
                    <div className="col-12">
                        <div className="text-center py-5 bg-light rounded-3 border border-dashed">
                            <div className="mb-3 text-muted opacity-50">
                                <Briefcase size={48} />
                            </div>
                            <h5 className="fw-bold text-muted">No jobs found</h5>
                            <p className="text-muted small">
                                {searchQuery ? "Try adjusting your search terms." : 
                                 statusFilter !== 'all' ? `No jobs matching '${statusFilter}' status.` :
                                 "Start building your library by adding a job."}
                            </p>
                            {(searchQuery || statusFilter !== 'all') ? (
                                <button className="btn btn-link text-primary" onClick={() => {setSearchQuery(''); setStatusFilter('all');}}>Clear Filters</button>
                            ) : (
                                <button className="btn btn-primary btn-sm mt-2" onClick={handleOpenAddModal}>+ Add Job</button>
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
                                        onViewApplication={handleViewApplication}
                                        onEdit={() => handleOpenEditModal(job.id)} 
                                        onDelete={() => handleOpenDeleteModal(job)}
                                        onViewDescription={handleViewDescription}
                                    />
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* --- MODALS --- */}
            
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

            {viewingDescJob && (
                <div className="modal show d-block" style={{backgroundColor: 'rgba(0,0,0,0.5)'}} tabIndex="-1" onClick={handleCloseDescModal}>
                    <div className="modal-dialog modal-lg modal-dialog-scrollable modal-dialog-centered" onClick={e => e.stopPropagation()}>
                        <div className="modal-content border-0 shadow-lg">
                            <div className="modal-header border-bottom-0 bg-light bg-opacity-50">
                                <div className="d-flex align-items-center gap-3">
                                    <div className="bg-white p-2 rounded border shadow-sm text-primary">
                                        <FileText size={20} />
                                    </div>
                                    <div>
                                        <h5 className="modal-title fw-bold text-dark">{viewingDescJob.title}</h5>
                                        <div className="text-muted small">{viewingDescJob.company}</div>
                                    </div>
                                </div>
                                <button type="button" className="btn-close" onClick={handleCloseDescModal}></button>
                            </div>
                            <div className="modal-body p-4 bg-white">
                                {viewingDescJob.displayed_description ? (
                                    <div 
                                        className="job-desc-content text-secondary"
                                        style={{ fontSize: '0.95rem', lineHeight: '1.6' }}
                                        dangerouslySetInnerHTML={{ __html: viewingDescJob.displayed_description }} 
                                    />
                                ) : (
                                    <div className="text-secondary" style={{ whiteSpace: 'pre-wrap', fontSize: '0.95rem', lineHeight: '1.6' }}>
                                        {viewingDescJob.description || "No description available."}
                                    </div>
                                )}
                            </div>
                            <div className="modal-footer border-top-0 bg-light bg-opacity-50">
                                <button type="button" className="btn btn-secondary" onClick={handleCloseDescModal}>Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default JobLibrary;