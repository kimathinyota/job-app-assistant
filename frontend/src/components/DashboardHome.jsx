import React from 'react';
import { 
    Plus, 
    FileText, 
    Briefcase, 
    Target, 
    ArrowRight, 
    TrendingUp, 
    Clock 
} from 'lucide-react';

const DashboardHome = ({ cvs, setActiveView }) => {
    const cvCount = cvs.length || 0;

    // Mock Data for "Executive" Feel
    // In a real implementation, pass these as props or fetch them
    const stats = [
        { label: 'Active Applications', value: '12', icon: Briefcase, change: '+2 this week', color: 'bg-blue-500' },
        { label: 'Interviews Scheduled', value: '3', icon: Clock, change: 'Next: Tomorrow 2PM', color: 'bg-emerald-500' },
        { label: 'Master CVs', value: cvCount, icon: FileText, change: 'Last updated 2d ago', color: 'bg-purple-500' },
        { label: 'Profile Completion', value: '85%', icon: TrendingUp, change: 'Add 2 more skills', color: 'bg-amber-500' },
    ];

    const recentActivity = [
        { id: 1, action: 'Applied to', target: 'Senior Frontend Engineer at Stripe', time: '2h ago' },
        { id: 2, action: 'Updated CV', target: 'Engineering_Lead_v2', time: '5h ago' },
        { id: 3, action: 'New Goal', target: 'Complete System Design Course', time: '1d ago' },
    ];

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-slate-900 dark:text-white tracking-tight">Dashboard</h1>
                    <p className="text-slate-500 dark:text-slate-400 mt-1">Welcome back. Here is your career overview.</p>
                </div>
                <div className="flex gap-3">
                    <button 
                        onClick={() => setActiveView('Application_Tracker')}
                        className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 rounded-lg text-sm font-medium hover:bg-slate-50 transition-colors shadow-sm"
                    >
                        <Plus size={16} /> New Application
                    </button>
                    <button 
                         onClick={() => setActiveView('CV_Manager')}
                         className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors shadow-sm"
                    >
                        <Plus size={16} /> Create CV
                    </button>
                </div>
            </div>

            {/* Bento Grid Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {stats.map((stat, index) => {
                    const Icon = stat.icon;
                    return (
                        <div key={index} className="bg-white dark:bg-slate-900 p-5 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md transition-shadow">
                            <div className="flex items-start justify-between mb-4">
                                <div className={`p-2 rounded-lg ${stat.color} bg-opacity-10`}>
                                    <Icon size={20} className={stat.color.replace('bg-', 'text-')} />
                                </div>
                                {index === 0 && <span className="flex h-2 w-2 rounded-full bg-blue-500"></span>}
                            </div>
                            <div className="text-2xl font-bold text-slate-900 dark:text-white mb-1">{stat.value}</div>
                            <div className="text-sm text-slate-500 dark:text-slate-400">{stat.label}</div>
                            <div className="mt-3 text-xs font-medium text-slate-400 dark:text-slate-500 flex items-center gap-1">
                                {stat.change}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Main Content Split */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* Quick Actions / Recent Activity */}
                <div className="lg:col-span-2 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="font-semibold text-slate-900 dark:text-white">Recent Activity</h3>
                        <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">View all</button>
                    </div>
                    <div className="space-y-4">
                        {recentActivity.map((item) => (
                            <div key={item.id} className="flex items-center justify-between p-3 hover:bg-slate-50 dark:hover:bg-slate-800/50 rounded-lg border border-transparent hover:border-slate-100 dark:hover:border-slate-800 transition-all">
                                <div className="flex items-center gap-4">
                                    <div className="h-8 w-8 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-500">
                                        <FileText size={14} />
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium text-slate-900 dark:text-white">
                                            {item.action} <span className="text-slate-600 dark:text-slate-300">{item.target}</span>
                                        </p>
                                    </div>
                                </div>
                                <span className="text-xs text-slate-400">{item.time}</span>
                            </div>
                        ))}
                        
                        {/* Empty State CTA if needed */}
                        {recentActivity.length === 0 && (
                            <div className="text-center py-8 text-slate-500">
                                No recent activity. Start by creating a CV.
                            </div>
                        )}
                    </div>
                </div>

                {/* Side Panel: Goals or Tips */}
                <div className="bg-slate-50 dark:bg-slate-800/30 rounded-xl border border-slate-200 dark:border-slate-800 p-6">
                    <h3 className="font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                        <Target size={18} className="text-blue-500" />
                        Weekly Focus
                    </h3>
                    <div className="space-y-3">
                        <div className="flex items-start gap-3">
                            <input type="checkbox" className="mt-1 rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
                            <span className="text-sm text-slate-600 dark:text-slate-300 line-through decoration-slate-400">Update Master CV Experience</span>
                        </div>
                        <div className="flex items-start gap-3">
                            <input type="checkbox" className="mt-1 rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
                            <span className="text-sm text-slate-600 dark:text-slate-300">Apply to 5 Senior Roles</span>
                        </div>
                        <div className="flex items-start gap-3">
                            <input type="checkbox" className="mt-1 rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
                            <span className="text-sm text-slate-600 dark:text-slate-300">Reach out to 3 Recruiters</span>
                        </div>
                    </div>
                    
                    <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
                        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Quick Links</h4>
                        <button onClick={() => setActiveView('Goal_Tracker')} className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium">
                            Go to Goal Tracker <ArrowRight size={14} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DashboardHome;