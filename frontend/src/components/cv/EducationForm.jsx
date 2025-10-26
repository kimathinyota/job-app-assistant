import React, { useState } from 'react';
import SkillLinker from './SkillLinker'; 

const EducationForm = ({ onSubmit, cvId, allSkills, onSkillCreate }) => {
    const [institution, setInstitution] = useState('');
    const [degree, setDegree] = useState('');
    const [field, setField] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [selectedSkillIds, setSelectedSkillIds] = useState([]); 

    const handleSubmit = (e) => {
        e.preventDefault();
        if (institution.trim() && degree.trim()) {
            onSubmit(cvId, { 
                institution, 
                degree, 
                field: field || 'N/A', 
                start_date: startDate || null, 
                end_date: endDate || null,
                skill_ids: selectedSkillIds 
            }, 'Education');
            setInstitution('');
            setDegree('');
            setField('');
            setStartDate('');
            setEndDate('');
            setSelectedSkillIds([]); // Reset selection
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ margin: '10px 0', padding: '15px', border: '1px solid #17a2b8', borderRadius: '5px', backgroundColor: '#e8f7fa', textAlign: 'left' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#17a2b8' }}>+ Add New Education</h4>
            
            <input type="text" value={institution} onChange={(e) => setInstitution(e.target.value)} placeholder="Institution Name" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            <input type="text" value={degree} onChange={(e) => setDegree(e.target.value)} placeholder="Degree/Certificate" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            <input type="text" value={field} onChange={(e) => setField(e.target.value)} placeholder="Field of Study (Optional)" style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />

            <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                <input type="text" value={startDate} onChange={(e) => setStartDate(e.target.value)} placeholder="Start Date (e.g., 2018)" style={{ flex: 1, padding: '8px' }} />
                <input type="text" value={endDate} onChange={(e) => setEndDate(e.target.value)} placeholder="End Date (e.g., Present)" style={{ flex: 1, padding: '8px' }} />
            </div>
            
            <SkillLinker 
                cvId={cvId}
                allSkills={allSkills} 
                selectedSkillIds={selectedSkillIds} 
                setSelectedSkillIds={setSelectedSkillIds}
                onCreateNewSkill={onSkillCreate} 
            />

            <button type="submit" style={{ backgroundColor: '#17a2b8', color: 'white', padding: '8px 15px', border: 'none', borderRadius: '4px', marginTop: '10px' }}>
                Create Education
            </button>
        </form>
    );
};

export default EducationForm;
