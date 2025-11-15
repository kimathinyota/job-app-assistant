// frontend/src/components/DashboardHome.jsx
import React, { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import { 
    Briefcase, 
    CheckCircle, 
    Clock, 
    MessageCircle, 
    Target,
    ListTodo,
    Zap,
    Layers,
    PlusCircle,
    AlertCircle,
    ArrowRight
} from 'lucide-react';

import { 
    fetchAllApplications, 
    fetchAllJobs 
} from '../api/applicationClient';

import { 
    fetchAllGoals, 
    fetchAllWorkItems 
} from '../api/goalClient'; 

import { fetchAllCVs } from '../api/cvClient'; // Import fetchAllCVs

// Component signature no longer accepts any props
const DashboardHome = () => {
    const navigate = useNavigate(); // Initialize navigate
    const [loading, setLoading] = useState(true);
    const [cvs, setCvs] = useState([]); // Add state for CVs
    const [data, setData] = useState({
        apps: [],
        jobs: [],
        goals: [],
        tasks: []
    });

    // --- 1. Data Loading (Now fetches CVs as well) ---
    useEffect(() => {
        const loadData = async () => {
            try {
                // Fetch all data in parallel, including CVs
                const [appsRes, jobsRes, goalsRes, tasksRes, cvsRes] = await Promise.all([
                    fetchAllApplications(),
                    fetchAllJobs(),
                    fetchAllGoals().catch(() => ({ data: [] })), 
                    fetchAllWorkItems().catch(() => ({ data: [] })),
                    fetchAllCVs() // Fetch CVs
                ]);
                
                const apps = appsRes.data || [];
                const jobs = jobsRes.data || [];
                const jobMap = new Map(jobs.map(j => [j.id, j]));

                // Enhance Apps with Job Data & Timestamps
                const enrichedApps = apps.map(app => ({
                    ...app, 
                    job: jobMap.get(app.job_id),
                    updatedAt: new Date(app.updated_at)
                })).sort((a, b) => b.updatedAt - a.updatedAt);

                setData({
                    apps: enrichedApps,
                    jobs: jobsRes.data || [],
                    goals: goalsRes.data || [],
                    tasks: tasksRes.data || []
                });
                setCvs(cvsRes || []); // Set CVs to local state

            } catch (err) {
                console.error("Dashboard load error:", err);
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, []); // Empty dependency array ensures this runs once on mount

    // --- 2. Smart Logic (Derived State) ---
    // This memo now correctly depends on the local 'cvs' state
    const stats = useMemo(() => {
        const now = new Date();
        const sevenDaysAgo = new Date(now.setDate(now.getDate() - 7));

        // Pipeline Stats
        const activeApps = data.apps.filter(a => a.status !== 'rejected');
        const interviews = data.apps.filter(a => a.status === 'interview');
        const offers = data.apps.filter(a => a.status === 'offer');
        const drafts = data.apps.filter(a => a.status === 'draft');
        const quiet = data.apps.filter(a => 
            (a.status === 'draft' || a.status === 'applied') && a.updatedAt < sevenDaysAgo
        );

        // Expiring Jobs Logic
        const upcomingDeadlines = drafts.filter(app => {
            if (!app.job?.application_end_date) return false;
            const deadline = new Date(app.job.application_end_date);
            const today = new Date();
            const diffTime = deadline - today;
            const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)); 
            return diffDays >= 0 && diffDays <= 3;
        });

        // CV Assets Logic: Find the "Master" (largest) CV
        const bestCV = cvs.length > 0 
            ? cvs.reduce((prev, current) => (prev.skills?.length > current.skills?.length) ? prev : current)
            : { skills: [], achievements: [], experiences: [], projects: [], education: [], hobbies: [] };

        return {
            activeCount: activeApps.length,
            interviewCount: interviews.length,
            offerCount: offers.length,
            quietCount: quiet.length,
            draftCount: drafts.length,
            
            interviews, 
            offers,
            quiet,
            upcomingDeadlines,

            // Asset Counts
            skillCount: bestCV.skills?.length || 0,
            achievementCount: bestCV.achievements?.length || 0,
            experienceCount: bestCV.experiences?.length || 0,
            projectCount: bestCV.projects?.length || 0,
            educationCount: bestCV.education?.length || 0,
            hobbyCount: bestCV.hobbies?.length || 0,
            
            masterCvName: bestCV.name || 'Master CV'
        };
    }, [data.apps, cvs]); // Now correctly depends on local state

    const topTasks = data.tasks.filter(t => t.status !== 'completed').slice(0, 5);
    const activeGoals = data.goals.filter(g => g.status === 'active').slice(0, 3);

    // --- 3. Dynamic Welcome Message Engine ---
    // (This function requires no changes as it depends on 'stats')
    const getWelcomeMessage = () => {
        // Priority 1: Interviews
        if (stats.interviewCount > 0) {
            const nextCompany = stats.interviews[0].job?.company || 'upcoming';
            return {
                title: "Good luck today.",
                text: `You have ${stats.interviewCount} active interview process${stats.interviewCount > 1 ? 'es' : ''}. Focus on your prep for ${nextCompany}.`
            };
        }
        // Priority 2: Offers
        if (stats.offerCount > 0) {
             return {
                title: "Congratulations!",
                text: `You have ${stats.offerCount} offer${stats.offerCount > 1 ? 's' : ''} waiting for your decision. Take your time to evaluate.`
            };
        }
        // Priority 3: Expiring Jobs (Deadlines)
        if (stats.upcomingDeadlines.length > 0) {
            const urgentApp = stats.upcomingDeadlines[0];
            return {
                title: "Heads up.",
                text: `The application for ${urgentApp.job?.title} closes soon. Might be worth finishing that draft today.`
            };
        }
        // Priority 4: Drafts / Momentum
        if (stats.draftCount > 0) {
            return {
                title: "Keep the momentum.",
                text: `You have ${stats.draftCount} application draft${stats.draftCount > 1 ? 's' : ''} in progress. Completing one today is a great win.`
            };
        }
        // Default
        return {
            title: "Welcome back.",
            text: "Your Career Operating System is active and ready. What would you like to achieve today?"
        };
    };

    const welcomeMsg = getWelcomeMessage();

    // --- 4. The "Smart Card" Renderer (Updated with navigate) ---
    const renderSmartCard = () => {
        if (stats.offerCount > 0) {
            return (
                <div className="card border-0 shadow-sm h-100 border-start border-4 border-success bg-success bg-opacity-10 hover-lift cursor-pointer" 
                     onClick={() => navigate('/applications')}> {/* <-- UPDATED */}
                    <div className="card-body p-4">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                            <span className="badge bg-success text-white">Victory Lap</span>
                            <CheckCircle size={20} className="text-success" />
                        </div>
                        <h3 className="h4 fw-bold text-success-emphasis mb-1">{stats.offerCount} Offer{stats.offerCount > 1 && 's'}!</h3>
                        <p className="small text-success-emphasis opacity-75 mb-0">Great work. Review details.</p>
                    </div>
                </div>
            );
        }
        if (stats.interviewCount > 0) {
            return (
                <div className="card border-0 shadow-sm h-100 border-start border-4 border-info bg-white hover-lift cursor-pointer" 
                     onClick={() => navigate(`/applications/${stats.interviews[0].id}`)}> {/* <-- UPDATED */}
                    <div className="card-body p-4">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                            <span className="badge bg-info bg-opacity-10 text-info-emphasis">High Priority</span>
                            <Clock size={20} className="text-info" />
                        </div>
                        <h3 className="h4 fw-bold text-dark mb-1">{stats.interviewCount} Interview{stats.interviewCount > 1 && 's'}</h3>
                        <p className="small text-muted mb-0">Prep mode active. Good luck!</p>
                        <div className="mt-2 text-info small d-flex align-items-center">
                            Open Prep Workspace <ArrowRight size={14} className="ms-1"/>
                        </div>
                    </div>
                </div>
            );
        }
        if (stats.upcomingDeadlines.length > 0) {
             return (
                <div className="card border-0 shadow-sm h-100 border-start border-4 border-danger bg-white hover-lift cursor-pointer" 
                     onClick={() => navigate(`/applications/${stats.upcomingDeadlines[0].id}`)}> {/* <-- UPDATED */}
                    <div className="card-body p-4">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                            <span className="badge bg-danger bg-opacity-10 text-danger-emphasis">Expiring Soon</span>
                            <AlertCircle size={20} className="text-danger" />
                        </div>
                        <h3 className="h4 fw-bold text-dark mb-1">{stats.upcomingDeadlines.length} Due Soon</h3>
                        <p className="small text-muted mb-0">Don't miss the window.</p>
                        <div className="mt-2 text-danger small d-flex align-items-center">
                            Finish Application <ArrowRight size={14} className="ms-1"/>
                        </div>
                    </div>
                </div>
            );
        }
        if (stats.quietCount > 0) {
            return (
                <div className="card border-0 shadow-sm h-100 border-start border-4 border-warning bg-white hover-lift cursor-pointer" 
                     onClick={() => navigate(`/applications/${stats.quiet[0].id}`)}> {/* <-- UPDATED */}
                    <div className="card-body p-4">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                            <span className="badge bg-warning bg-opacity-10 text-warning-emphasis">Nudge Needed</span>
                            <MessageCircle size={20} className="text-warning" />
                        </div>
                        <h3 className="h4 fw-bold text-dark mb-1">{stats.quietCount} Quiet App{stats.quietCount > 1 && 's'}</h3>
                        <p className="small text-muted mb-0">No updates in 7 days.</p>
                        <div className="mt-2 text-warning small d-flex align-items-center">
                            Send Follow-up <ArrowRight size={14} className="ms-1"/>
                        </div>
                    </div>
                </div>
            );
        }

        const topGoal = activeGoals[0];
        return (
            <div className="card border-0 shadow-sm h-100 border-start border-4 border-primary bg-primary bg-opacity-10 hover-lift cursor-pointer" 
                 onClick={() => navigate('/goals')}> {/* <-- UPDATED */}
                <div className="card-body p-4">
                    <div className="d-flex justify-content-between align-items-center mb-2">
                        <span className="badge bg-primary text-white">Current Focus</span>
                        <Target size={20} className="text-primary" />
                    </div>
                    {topGoal ? (
                        <>
                            <h3 className="h5 fw-bold text-primary-emphasis mb-1">{topGoal.title}</h3>
                            <div className="progress mt-2" style={{height: '6px'}}>
                                <div className="progress-bar bg-primary" style={{width: `${(topGoal.progress || 0) * 100}%`}}></div>
                            </div>
                            <p className="small text-primary-emphasis opacity-75 mt-2 mb-0">Keep pushing forward.</p>
                        </>
                    ) : (
                        <>
                            <h3 className="h4 fw-bold text-primary-emphasis mb-1">Ready to Grow?</h3>
                            <p className="small text-primary-emphasis opacity-75 mb-0">Set a goal to track your progress.</p>
                            <button className="btn btn-sm btn-primary mt-2">+ Set Goal</button>
                        </>
                    )}
                </div>
            </div>
        );
    };

    if (loading) return <div className="p-5 text-center text-muted">Initializing Command Center...</div>;

    // Calculate total assets count
    const totalAssets = stats.skillCount + stats.achievementCount + stats.experienceCount + stats.projectCount + stats.educationCount + stats.hobbyCount;

    // --- Helper for Deep Linking to CV Manager ---
    const handleNavigateToCVSection = (sectionName) => {
        // Navigate to /cv and pass the sectionName in location.state
        navigate('/cv', { state: { initialSection: sectionName } });
    };

    return (
        <div className="container-fluid px-0 pb-5">
            <style>
                {`
                .hover-lift {
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }
                .hover-lift:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
                }
                .cursor-pointer {
                    cursor: pointer;
                }
                `}
            </style>
            
            {/* --- 1. Header --- */}
            <div className="mb-4">
                <h1 className="h3 fw-bold text-dark">{welcomeMsg.title}</h1>
                <p className="text-muted">{welcomeMsg.text}</p>
            </div>

            {/* --- 2. Dynamic Deck (The Top Row) --- */}
            <div className="row g-4 mb-4">
                
                {/* Card 1: The Engine (Deep Asset View) */}
                <div className="col-md-4">
                    <div className="card border-0 shadow-sm h-100 bg-white hover-lift cursor-pointer" 
                         onClick={() => navigate('/cv')}> {/* <-- UPDATED */}
                        <div className="card-body p-4">
                            <div className="d-flex align-items-center gap-3 mb-3">
                                <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
                                    <Layers size={24} className="text-primary" />
                                </div>
                                <div>
                                    <h6 className="fw-bold text-muted text-uppercase small mb-0">Career Assets</h6>
                                    <div className="small text-muted">{stats.masterCvName}</div>
                                </div>
                            </div>
                            <div className="d-flex align-items-end gap-2 mb-3">
                                <h2 className="display-6 fw-bold mb-0">{totalAssets}</h2>
                                <span className="text-muted mb-2 small">Total Data Points</span>
                            </div>
                            
                            {/* Asset Micro-Grid */}
                            <div className="d-flex flex-wrap gap-2">
                                <span className="badge bg-light text-dark border fw-normal">{stats.skillCount} Skills</span>
                                <span className="badge bg-light text-dark border fw-normal">{stats.achievementCount} Wins</span>
                                <span className="badge bg-light text-dark border fw-normal">{stats.experienceCount} Exp</span>
                                <span className="badge bg-light text-dark border fw-normal">{stats.projectCount} Proj</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Card 2: The Pursuit (Pipeline) */}
                <div className="col-md-4">
                    <div className="card border-0 shadow-sm h-100 bg-white hover-lift cursor-pointer" 
                         onClick={() => navigate('/applications')}> {/* <-- UPDATED */}
                        <div className="card-body p-4">
                            <div className="d-flex align-items-center gap-3 mb-3">
                                <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
                                    <Briefcase size={24} className="text-dark opacity-75" />
                                </div>
                                <div>
                                    <h6 className="fw-bold text-muted text-uppercase small mb-0">Active Pursuits</h6>
                                    <div className="small text-muted">In the pipeline</div>
                                </div>
                            </div>
                            <div className="d-flex align-items-end gap-2">
                                <h2 className="display-6 fw-bold mb-0">{stats.activeCount}</h2>
                                <span className="text-muted mb-2 small">Applications</span>
                            </div>
                            <div className="mt-3">
                                {stats.activeCount === 0 ? (
                                    <span className="text-muted small fst-italic">Ready to start applying?</span>
                                ) : (
                                    <span className="text-success small fw-medium flex items-center gap-1">
                                        <Zap size={12}/> System Active
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Card 3: The Context (Smart Slot) */}
                <div className="col-md-4">
                    {renderSmartCard()}
                </div>
            </div>

            {/* --- 3. Workbench (Refined Quick Actions) --- */}
            <div className="card border-0 shadow-sm mb-5 bg-light">
                <div className="card-body py-3 px-4 d-flex align-items-center justify-content-between flex-wrap gap-3">
                    <span className="fw-bold text-muted small text-uppercase">Quick Workbench</span>
                    <div className="d-flex gap-2">
                        {/* Add Skill -> CV Manager (Deep Link) */}
                        <button onClick={() => handleNavigateToCVSection('Skills')} className="btn btn-white bg-white border shadow-sm btn-sm fw-medium text-dark d-flex align-items-center gap-2 hover-lift"> {/* <-- UPDATED */}
                            <PlusCircle size={14} className="text-primary"/> Add Skill
                        </button>
                        {/* Add Experience -> CV Manager (Deep Link) */}
                        <button onClick={() => handleNavigateToCVSection('Experiences')} className="btn btn-white bg-white border shadow-sm btn-sm fw-medium text-dark d-flex align-items-center gap-2 hover-lift"> {/* <-- UPDATED */}
                            <PlusCircle size={14} className="text-primary"/> Add Experience
                        </button>
                        <div className="vr opacity-25 mx-1"></div>
                        {/* Track Job -> Tracker */}
                        <button onClick={() => navigate('/applications')} className="btn btn-primary btn-sm fw-medium d-flex align-items-center gap-2 hover-lift"> {/* <-- UPDATED */}
                            <Briefcase size={14}/> Track Job
                        </button>
                    </div>
                </div>
            </div>

            {/* --- 4. Main Content Grid --- */}
            <div className="row g-5">
                
                {/* LEFT: Application Journey */}
                <div className="col-lg-7">
                    <div className="d-flex justify-content-between align-items-center mb-3">
                        <h5 className="fw-bold text-dark mb-0">Recent Journey</h5>
                        <button onClick={() => navigate('/applications')} className="btn btn-link btn-sm text-decoration-none">View All</button> {/* <-- UPDATED */}
                    </div>
                    
                    <div className="card border-0 shadow-sm">
                        <div className="list-group list-group-flush">
                            {data.apps.slice(0, 5).map(app => (
                                <div 
                                    key={app.id} 
                                    onClick={() => navigate(`/applications/${app.id}`)} // <-- UPDATED
                                    className="list-group-item list-group-item-action p-3 border-bottom d-flex align-items-center justify-content-between cursor-pointer hover-lift"
                                >
                                    <div className="d-flex align-items-center gap-3">
                                        <div className="bg-light rounded p-2 d-flex align-items-center justify-content-center" style={{width: '40px', height: '40px'}}>
                                            <span className="fw-bold text-muted">{app.job?.company?.charAt(0) || '?'}</span>
                                        </div>
                                        <div>
                                            <div className="fw-bold text-dark">{app.job?.title}</div>
                                            <div className="small text-muted">{app.job?.company} &bull; <span className="text-lowercase">{app.updatedAt.toLocaleDateString(undefined, {month:'short', day:'numeric'})}</span></div>
                                        </div>
                                    </div>
                                    <div className="d-flex align-items-center gap-3">
                                        <span className={`badge rounded-pill fw-normal px-3 py-2 
                                            ${app.status === 'interview' ? 'bg-info bg-opacity-10 text-info-emphasis' : 
                                              app.status === 'offer' ? 'bg-success bg-opacity-10 text-success-emphasis' : 
                                              'bg-light text-secondary'}`}>
                                            {app.status}
                                        </span>
                                    </div>
                                </div>
                            ))}
                            {data.apps.length === 0 && (
                                <div className="p-5 text-center text-muted">
                                    <Briefcase size={32} className="mb-3 opacity-25"/>
                                    <p>Your pipeline is empty. Use the workbench to track a job!</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* RIGHT: Action Center (Goals & Tasks) */}
                <div className="col-lg-5">
                    <div className="d-flex justify-content-between align-items-center mb-3">
                        <h5 className="fw-bold text-dark mb-0">Action Center</h5>
                        <button onClick={() => navigate('/goals')} className="btn btn-link btn-sm text-decoration-none">Open Tracker</button> {/* <-- UPDATED */}
                    </div>

                    {/* Goals Section */}
                    {activeGoals.length > 0 && (
                        <div className="card border-0 shadow-sm mb-4 bg-gradient-to-br from-white to-slate-50">
                            <div className="card-body p-3">
                                <h6 className="small fw-bold text-uppercase text-muted mb-3 d-flex align-items-center gap-2">
                                    <Target size={14}/> Active Goals
                                </h6>
                                <div className="d-flex flex-column gap-3">
                                    {activeGoals.map(goal => (
                                        <div key={goal.id}>
                                            <div className="d-flex justify-content-between small mb-1">
                                                <span className="fw-medium">{goal.title}</span>
                                                <span className="text-muted">{Math.round((goal.progress || 0) * 100)}%</span>
                                            </div>
                                            <div className="progress" style={{height: '4px'}}>
                                                <div className="progress-bar bg-primary" style={{width: `${(goal.progress || 0) * 100}%`}}></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Tasks Section */}
                    <div className="card border-0 shadow-sm">
                        <div className="card-header bg-white border-0 pt-3 pb-0">
                            <h6 className="small fw-bold text-uppercase text-muted mb-0 d-flex align-items-center gap-2">
                                <ListTodo size={14}/> Next Steps
                            </h6>
                        </div>
                        <div className="card-body p-0">
                            <ul className="list-group list-group-flush">
                                {topTasks.length > 0 ? topTasks.map(task => (
                                    <li key={task.id} className="list-group-item border-0 py-3 d-flex gap-3">
                                        <div className="mt-1">
                                            <div className={`rounded-circle border ${task.type === 'application' ? 'border-primary' : 'border-secondary'}`} style={{width: '16px', height: '16px'}}></div>
                                        </div>
                                        <div>
                                            <div className="fw-medium text-dark small">{task.title}</div>
                                            {task.related_job_id && <div className="text-xs text-muted">Related to Job App</div>}
                                        </div>
                                    </li>
                                )) : (
                                    <li className="list-group-item border-0 py-4 text-center text-muted small">
                                        No pending tasks. <br/>
                                        <button onClick={() => navigate('/goals')} className="btn btn-link btn-sm p-0">Create a task</button> {/* <-- UPDATED */}
                                    </li>
                                )}
                            </ul>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};

export default DashboardHome;