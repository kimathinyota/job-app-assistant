// frontend/src/components/applications/ApplicationCard.jsx
import React from 'react';
import { Building2, Clock, Calendar, Trash2 } from 'lucide-react';

const ApplicationCard = ({ application, job, onClick, onDelete }) => {
  
  const handleDeleteClick = (e) => {
    e.stopPropagation(); 
    onDelete();
  };

  // Format Date
  const dateStr = new Date(application.updated_at).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric'
  });

  // Status Color Logic (Border Accent)
  const getStatusColor = (status) => {
    switch(status) {
      case 'offer': return 'border-success';
      case 'interview': return 'border-info';
      case 'rejected': return 'border-danger';
      case 'applied': return 'border-primary';
      default: return 'border-secondary';
    }
  };

  return (
    <div 
      className={`card border-0 shadow-sm mb-3 hover-lift cursor-pointer border-start border-4 ${getStatusColor(application.status)}`}
      onClick={onClick}
    >
      <div className="card-body p-3">
        <div className="d-flex justify-content-between align-items-start mb-2">
          
          {/* Company Logo / Title Section */}
          <div className="d-flex align-items-center gap-3">
            {/* Logo Placeholder */}
            <div className="rounded-3 d-flex align-items-center justify-content-center bg-light text-muted fw-bold border" 
                 style={{ width: '40px', height: '40px', fontSize: '18px' }}>
              {job.company ? job.company.charAt(0).toUpperCase() : <Building2 size={20}/>}
            </div>
            
            <div>
              <h6 className="fw-bold text-dark mb-0 text-truncate" style={{maxWidth: '160px'}}>
                {job.title || 'Untitled Position'}
              </h6>
              <div className="small text-muted d-flex align-items-center gap-1">
                {job.company || 'Unknown Company'}
              </div>
            </div>
          </div>

          {/* Delete Action (Top Right) */}
          <button 
            onClick={handleDeleteClick}
            className="btn btn-link text-muted p-0 opacity-50 hover-opacity-100"
            title="Delete Application"
          >
            <Trash2 size={16} />
          </button>
        </div>

        {/* Footer Info */}
        <div className="d-flex align-items-center justify-content-between mt-3 pt-2 border-top border-light">
          <span className="badge bg-light text-secondary fw-normal border">
             {application.status}
          </span>
          <div className="small text-muted d-flex align-items-center gap-1">
            <Clock size={12} /> {dateStr}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApplicationCard;