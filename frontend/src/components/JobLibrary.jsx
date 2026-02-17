// frontend/src/pages/JobLibrary.js
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom'; 
import { 
    Plus, Search, Briefcase, XCircle, Filter, 
    CheckCircle2, CircleDashed, ArrowUpDown, Sparkles, Loader2, FileText 
} from 'lucide-react';
import { debounce } from 'lodash'; 

import { 
    fetchAllJobs, 
    fetchAllApplications, 
    createMapping,
    createApplication,
    deleteJob,
    scoreAllJobs 
} from '../api/applicationClient';
import { fetchAllCVs } from '../api/cvClient';
import { getCurrentUser } from '../api/authClient'; 

// Components
import JobCard from './applications/JobCard';
import JobModal from './applications/JobModal'; 
import DeleteConfirmationModal from './common/DeleteConfirmationModal';
// [1] IMPORT THE NEW MODAL
import JobPreviewModal from './applications/JobPreviewModal'; 

const JobLibrary = () => {
    const navigate = useNavigate();

    // --- Data State ---
    const [loading, setLoading] = useState(true);
    const [analyzing, setAnalyzing] = useState(false);
    const [jobs, setJobs] = useState([]);
    const [applications, setApplications] = useState([]);
    const [cvs, setCvs] = useState([]);
    const [user, setUser] = useState(null);

    // --- UI States ---
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState('all'); 
    const [sortBy, setSortBy] = useState('recommended'); 

    // --- Modal States ---
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalJobId, setModalJobId] = useState(null); 
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [itemToDelete, setItemToDelete] = useState(null);

    // [2] ADD STATE FOR PREVIEW MODAL
    const [previewJob, setPreviewJob] = useState(null); 

    // --- 1. Load Data ---
    const loadData = useCallback(async (query = '', sort = 'recommended') => {
        setLoading(true);
        try {
            const [userRes, cvsRes, appsRes, jobsRes] = await Promise.all([
                getCurrentUser(), 
                fetchAllCVs(), 
                fetchAllApplications(),
                fetchAllJobs({ q: query, sort: sort }), 
            ]);

            setUser(userRes.data || userRes); 
            setJobs(jobsRes.data || []);
            setApplications(appsRes.data || []);
            setCvs(cvsRes || []); 

        } catch (err) {
            console.error("Failed to load library", err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        const debouncedLoad = debounce(() => loadData(searchQuery, sortBy), 500);
        debouncedLoad();
        return () => debouncedLoad.cancel();
    }, [searchQuery, sortBy, loadData]);


    // --- 2. Computed Data ---
    const defaultCvId = useMemo(() => {
        if (user?.primary_cv_id) return user.primary_cv_id;
        return cvs?.[0]?.id || null;
    }, [user, cvs]);

    const targetCvName = useMemo(() => {
        if (!defaultCvId) return null;
        const found = cvs.find(c => c.id === defaultCvId);
        return found ? found.name : 'Unknown CV';
    }, [cvs, defaultCvId]);

    const applicationMap = useMemo(() => {
        const map = new Map();
        applications.forEach(app => map.set(app.job_id, app));
        return map;
    }, [applications]);

    const counts = useMemo(() => {
        const pending = jobs.filter(j => !applicationMap.has(j.id)).length;
        const started = jobs.filter(j => applicationMap.has(j.id)).length;
        return { all: jobs.length, pending, started };
    }, [jobs, applicationMap]);

    const displayJobs = useMemo(() => {
        let filtered = jobs;
        if (statusFilter === 'pending') filtered = jobs.filter(j => !applicationMap.has(j.id));
        if (statusFilter === 'started') filtered = jobs.filter(j => applicationMap.has(j.id));
        return filtered;
    }, [jobs, statusFilter, applicationMap]);

    // --- 3. Handlers ---
    const handleRunAnalysis = async () => {
        if (!defaultCvId) return alert("Please upload a CV first.");
        setAnalyzing(true);
        try {
            await scoreAllJobs(defaultCvId);
            await loadData(searchQuery, sortBy);
        } catch (err) {
            alert("Analysis failed.");
        } finally {
            setAnalyzing(false);
        }
    };

    const handleOpenAddModal = () => { setModalJobId(null); setIsModalOpen(true); };
    const handleOpenEditModal = (jobId) => { setModalJobId(jobId); setIsModalOpen(true); };
    
    // [3] UPDATE HANDLER: Open the Preview Modal instead of the Edit Modal
    const handleViewDescription = (job) => {
        setPreviewJob(job);
    };

    const handleCloseModal = () => { setIsModalOpen(false); setModalJobId(null); };
    const handleJobUpdated = () => { loadData(searchQuery, sortBy); };
    
    const handleOpenDeleteModal = (job) => { setItemToDelete(job); setIsDeleteModalOpen(true); };
    const handleCloseDeleteModal = () => { setItemToDelete(null); setIsDeleteModalOpen(false); };

    const handleConfirmDelete = async () => {
        if (!itemToDelete) return;
        try {
            await deleteJob(itemToDelete.id);
            handleCloseDeleteModal();
            loadData(searchQuery, sortBy); 
        } catch (err) {
            alert(`Failed to delete job: ${err.message}`);
        }
    };

    const handleStartApplication = async (jobId, baseCvId) => {
        if (!baseCvId) return alert("Please select a CV to start.");
        try {
            const mappingRes = await createMapping(jobId, baseCvId);
            const appRes = await createApplication(jobId, baseCvId, mappingRes.data.id);
            navigate(`/application/${appRes.data.id}`);
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="container-fluid px-0 pb-5">
             <style>
                {`
                .hover-lift { transition: transform 0.2s ease, box-shadow 0.2s ease; }
                .hover-lift:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0,0,0,0.08) !important; }
                .nav-pills .nav-link { color: #64748b; font-weight: 500; font-size: 0.9rem; padding: 0.5rem 1rem; border-radius: 6px; transition: all 0.2s; }
                .nav-pills .nav-link:hover { background-color: #f1f5f9; color: #0f172a; }
                .nav-pills .nav-link.active { background-color: white; color: var(--bs-primary); box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-weight: 600; }
                .search-input:focus { box-shadow: none; border-color: var(--bs-primary); }
                `}
            </style>

            {/* --- HEADER --- */}
            <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-4 gap-3">
                <div className="d-flex align-items-center gap-3">
                    <div className="bg-primary bg-opacity-10 p-3 rounded-3 text-primary">
                        <Briefcase size={24} />
                    </div>
                    <div>
                        <h2 className="fw-bold text-dark mb-1 h4">Job Library</h2>
                        <div className="d-flex align-items-center gap-2 text-muted small">
                            {defaultCvId ? (
                                <span className="d-flex align-items-center gap-1 bg-light px-2 py-1 rounded border">
                                    <FileText size={12}/> 
                                    Targeting: <strong className="text-dark">{targetCvName}</strong>
                                </span>
                            ) : (
                                <span>Manage your job opportunities</span>
                            )}
                        </div>
                    </div>
                </div>
                <div>
                    <button className="btn btn-primary d-flex align-items-center gap-2 shadow-sm hover-lift px-4 py-2" onClick={handleOpenAddModal}>
                        <Plus size={18} strokeWidth={2.5}/> <span className="fw-medium">Add Job</span>
                    </button>
                </div>
            </div>

            {/* --- TOOLBAR --- */}
            <div className="card border-0 shadow-sm mb-4 bg-light bg-opacity-50">
                <div className="card-body p-2">
                    <div className="d-flex flex-column flex-lg-row gap-3 align-items-center justify-content-between">
                        {/* Tabs */}
                        <div className="nav nav-pills bg-light rounded p-1 border d-inline-flex w-100 w-lg-auto justify-content-center justify-content-lg-start">
                            <button className={`nav-link d-flex align-items-center gap-2 ${statusFilter === 'all' ? 'active' : ''}`} onClick={() => setStatusFilter('all')}>
                                All Jobs <span className="badge bg-secondary bg-opacity-10 text-secondary rounded-pill">{counts.all}</span>
                            </button>
                            <button className={`nav-link d-flex align-items-center gap-2 ${statusFilter === 'pending' ? 'active' : ''}`} onClick={() => setStatusFilter('pending')}>
                                To Apply <span className="badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-25 rounded-pill">{counts.pending}</span>
                            </button>
                            <button className={`nav-link d-flex align-items-center gap-2 ${statusFilter === 'started' ? 'active' : ''}`} onClick={() => setStatusFilter('started')}>
                                Applied <span className="badge bg-success bg-opacity-10 text-success border border-success border-opacity-25 rounded-pill">{counts.started}</span>
                            </button>
                        </div>

                        {/* Search & Actions */}
                        <div className="d-flex gap-2 w-100 w-lg-auto">
                            <div className="input-group shadow-sm" style={{maxWidth: '300px'}}>
                                <span className="input-group-text bg-white border-end-0 text-muted ps-3"><Search size={16}/></span>
                                <input type="text" className="form-control border-start-0 ps-0 search-input py-2" placeholder="Search..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}/>
                                {searchQuery && <button className="btn btn-white border border-start-0" onClick={() => setSearchQuery('')}><XCircle size={14} className="text-muted"/></button>}
                            </div>

                            <div className="dropdown">
                                <button className="btn btn-white bg-white border shadow-sm d-flex align-items-center gap-2 text-muted h-100 px-3" type="button" data-bs-toggle="dropdown">
                                    <ArrowUpDown size={16}/> <span className="d-none d-md-inline small fw-medium">Sort</span>
                                </button>
                                <ul className="dropdown-menu dropdown-menu-end shadow-sm border-0">
                                    <li><button className="dropdown-item" onClick={() => setSortBy('recommended')}>✨ Recommended</button></li>
                                    <li><button className="dropdown-item" onClick={() => setSortBy('score')}>🎯 Best Score</button></li>
                                    <li><button className="dropdown-item" onClick={() => setSortBy('deadline')}>⏰ Closing Soon</button></li>
                                    <li><button className="dropdown-item" onClick={() => setSortBy('date')}>📅 Newest</button></li>
                                </ul>
                            </div>

                            <button onClick={handleRunAnalysis} disabled={analyzing} className="btn btn-white bg-white border shadow-sm d-flex align-items-center gap-2 text-primary hover-lift px-3">
                                {analyzing ? <Loader2 size={18} className="animate-spin"/> : <Sparkles size={18}/>}
                                <span className="d-none d-md-inline fw-medium">{analyzing ? 'Scanning...' : 'Scan'}</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* --- SECTION 3: GRID --- */}
            <div className="row g-4">
                {loading ? (
                    <div className="text-center py-5 w-100"><div className="spinner-border text-primary" role="status"/></div>
                ) : displayJobs.length === 0 ? (
                    <div className="col-12 text-center py-5 bg-white rounded-3 border border-dashed shadow-sm">
                        <div className="mb-3 bg-light rounded-circle p-4 d-inline-block"><Briefcase size={48} className="text-muted opacity-50" /></div>
                        <h5 className="fw-bold text-dark mb-1">No jobs found</h5>
                        <p className="text-muted small mb-3">Try adjusting your search or add a new job.</p>
                        {searchQuery ? (
                             <button className="btn btn-outline-secondary btn-sm" onClick={() => setSearchQuery('')}>Clear Search</button>
                        ) : (
                             <button className="btn btn-primary btn-sm" onClick={handleOpenAddModal}>Add Job</button>
                        )}
                    </div>
                ) : (
                    displayJobs.map(job => {
                        const application = applicationMap.get(job.id);
                        const displayScore = application ? application.match_score : job.match_score;
                        const displayBadges = application ? application.cached_badges : job.cached_badges;

                        return (
                            <div key={job.id} className="col-12 col-md-6 col-xl-4 d-flex align-items-stretch">
                                <div className="w-100 h-100 hover-lift transition-all">
                                    <JobCard
                                        job={job}
                                        cvs={cvs}
                                        defaultCvId={defaultCvId}
                                        application={application}
                                        matchScore={displayScore}
                                        badges={displayBadges}
                                        onStartApplication={handleStartApplication}
                                        onViewApplication={(appId) => navigate(`/application/${appId}`)}
                                        onEdit={() => handleOpenEditModal(job.id)} 
                                        onDelete={() => handleOpenDeleteModal(job)}
                                        
                                        // Connected to the new Handler
                                        onViewDescription={handleViewDescription} 
                                    />
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* --- MODALS --- */}
            
            {/* The existing Edit/Add Modal */}
            <JobModal
                key={modalJobId || 'new'} 
                initialJobId={modalJobId}
                isOpen={isModalOpen}
                onClose={handleCloseModal}
                onJobUpdated={handleJobUpdated}
            />

            {/* [4] RENDER THE NEW PREVIEW MODAL */}
            <JobPreviewModal 
                job={previewJob} 
                isOpen={!!previewJob} 
                onClose={() => setPreviewJob(null)} 
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

export default JobLibrary;