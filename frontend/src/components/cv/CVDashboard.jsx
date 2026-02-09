import React from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { Briefcase, BookOpen, Cpu, Layers, Award, Smile } from 'lucide-react';

const SectionCard = ({ title, count, icon: Icon, colorClass, onClick }) => (
    <div onClick={onClick} className="col-md-4 mb-3">
        <div className="card border-0 shadow-sm h-100 hover-lift cursor-pointer transition-all">
            <div className="card-body d-flex align-items-center gap-3 p-4">
                <div className={`p-3 rounded-circle bg-opacity-10 ${colorClass.replace('text-', 'bg-')}`}>
                    <Icon size={24} className={colorClass} />
                </div>
                <div>
                    <h5 className="fw-bold text-dark mb-0">{title}</h5>
                    <span className="text-muted small">{count} items</span>
                </div>
            </div>
        </div>
    </div>
);

const CVDashboard = () => {
    const { cv } = useOutletContext();
    const navigate = useNavigate();

    return (
        <div>
            <style>{`.hover-lift:hover { transform: translateY(-4px); box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important; } .cursor-pointer { cursor: pointer; }`}</style>
            
            <div className="row g-3 py-2">
                <SectionCard 
                    title="Experience" 
                    count={cv.experiences?.length || 0} 
                    icon={Briefcase} 
                    colorClass="text-blue-600" 
                    onClick={() => navigate('experience')} 
                />
                <SectionCard 
                    title="Education" 
                    count={cv.education?.length || 0} 
                    icon={BookOpen} 
                    colorClass="text-indigo-600" 
                    onClick={() => navigate('education')} 
                />
                <SectionCard 
                    title="Projects" 
                    count={cv.projects?.length || 0} 
                    icon={Cpu} 
                    colorClass="text-purple-600" 
                    onClick={() => navigate('projects')} 
                />
                <SectionCard 
                    title="Master Skills" 
                    count={cv.skills?.length || 0} 
                    icon={Layers} 
                    colorClass="text-emerald-600" 
                    onClick={() => navigate('skills')} 
                />
                <SectionCard 
                    title="Achievements" 
                    count={cv.achievements?.length || 0} 
                    icon={Award} 
                    colorClass="text-amber-500" 
                    onClick={() => navigate('achievements')} 
                />
                <SectionCard 
                    title="Hobbies" 
                    count={cv.hobbies?.length || 0} 
                    icon={Smile} 
                    colorClass="text-pink-500" 
                    onClick={() => navigate('hobbies')} 
                />
            </div>
        </div>
    );
};

export default CVDashboard;