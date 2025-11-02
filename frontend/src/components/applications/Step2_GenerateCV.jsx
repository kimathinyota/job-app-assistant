// frontend/src/components/applications/Step2_GenerateCV.jsx
import React, { useState, useMemo } from 'react';
import { generateCvPrompt } from '../../api/applicationClient';
import PromptModal from './PromptModal';

const Step2_GenerateCV = ({ job, cv, mapping, onPrev, onNext }) => {
    const [cvPromptJson, setCvPromptJson] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    // Calculate the "Derived CV" components for display
    const derivedCV = useMemo(() => {
        const mappedExpIds = new Set(mapping.pairs.map(p => p.experience_id));
        const mappedExperiences = cv.experiences.filter(exp => mappedExpIds.has(exp.id));
        
        let skillIds = new Set();
        mappedExperiences.forEach(exp => {
            (exp.skill_ids || []).forEach(id => skillIds.add(id));
        });

        const mappedSkills = cv.skills.filter(s => skillIds.has(s.id));
        
        return { mappedExperiences, mappedSkills };
    }, [cv, mapping]);


    const handleGeneratePrompt = async () => {
        setIsLoading(true);
        try {
            const res = await generateCvPrompt(cv.id, job.id);
            setCvPromptJson(JSON.stringify(res.data, null, 2));
            setIsModalOpen(true);
        } catch (err) {
            alert("Failed to generate prompt.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div>
            <h4 className="h5">Step 2: Review Your Tailored CV</h4>
            <p className="text-muted">This is the content that will be sent to the AI, based on your mapping.</p>
            
            <div className="border p-3 rounded bg-light" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                <h5>Mapped Experiences ({derivedCV.mappedExperiences.length})</h5>
                {derivedCV.mappedExperiences.map(exp => (
                    <div key={exp.id} className="mb-2">
                        <strong>{exp.title} @ {exp.company}</strong>
                    </div>
                ))}
                
                <h5 className="mt-3">Related Skills ({derivedCV.mappedSkills.length})</h5>
                <div className="d-flex flex-wrap gap-1">
                    {derivedCV.mappedSkills.map(skill => (
                        <span key={skill.id} className="badge bg-secondary">
                            {skill.name}
                        </span>
                    ))}
                </div>
            </div>

            <button 
                className="btn btn-info mt-3" 
                onClick={handleGeneratePrompt}
                disabled={isLoading}
            >
                {isLoading ? "Generating..." : "Generate CV Prompt"}
            </button>

            <PromptModal
                isOpen={isModalOpen}
                jsonString={cvPromptJson}
                onClose={() => setIsModalOpen(false)}
            />
            
            <div className="d-flex justify-content-between mt-4">
                <button className="btn btn-secondary" onClick={onPrev}>
                    &lt; Back: Mapping
                </button>
                <button className="btn btn-primary" onClick={onNext}>
                    Next: Build Cover Letter &gt;
                </button>
            </div>
        </div>
    );
};

export default Step2_GenerateCV;