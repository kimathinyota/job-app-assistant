// frontend/src/components/applications/ApplicationsView.jsx
import React, { useState, useEffect } from 'react';
import { fetchAllApplications, fetchAllJobs } from '../../api/applicationClient';
import ApplicationCard from './ApplicationCard';

const ApplicationsView = ({ onNavigateToWorkspace }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [data, setData] = useState({ applications: [], jobs: [] });

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [appsRes, jobsRes] = await Promise.all([
                fetchAllApplications(),
                fetchAllJobs()
            ]);
            setData({ applications: appsRes.data || [], jobs: jobsRes.data || [] });
        } catch (err) {
            setError("Failed to load applications.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    const getJob = (jobId) => data.jobs.find(j => j.id === jobId) || {};

    const columns = {
        draft: { title: 'Drafts', apps: data.applications.filter(a => a.status === 'draft') },
        applied: { title: 'Applied', apps: data.applications.filter(a => a.status === 'applied') },
        interview: { title: 'Interviewing', apps: data.applications.filter(a => a.status === 'interview') },
        offer: { title: 'Offers', apps: data.applications.filter(a => a.status === 'offer') },
        rejected: { title: 'Rejected', apps: data.applications.filter(a => a.status === 'rejected') },
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
                                col.apps.map(app => (
                                    <ApplicationCard
                                        key={app.id}
                                        application={app}
                                        job={getJob(app.job_id)}
                                        onClick={() => onNavigateToWorkspace(app.id)}
                                    />
                                ))
                            )}
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default ApplicationsView;