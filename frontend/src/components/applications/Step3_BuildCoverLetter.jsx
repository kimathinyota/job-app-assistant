// frontend/src/components/applications/Step3_BuildCoverLetter.jsx
import React, { useState, useEffect } from 'react';
import { 
    generateCoverLetterPrompt, 
    createCoverLetter,
    addCoverLetterIdea,
    addCoverLetterParagraph
} from '../../api/applicationClient';
import PromptModal from './PromptModal';

const Step3_BuildCoverLetter = ({ application, mapping, onPrev, onNext }) => {
    const [coverLetter, setCoverLetter] = useState(null);
    const [clPromptJson, setClPromptJson] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    // This is a simplified UI. A real one would be more complex.
    // We'll create a default CoverLetter and add one idea/paragraph.
    
    useEffect(() => {
        // Check if a CoverLetter already exists or create one
        // This is simplified; in a real app, you'd fetch or create
        const setupCoverLetter = async () => {
            if (!application.cover_letter_id) {
                try {
                    // Create a new CoverLetter
                    const clRes = await createCoverLetter(application.job_id, application.base_cv_id, application.mapping_id);
                    setCoverLetter(clRes.data);
                    // This is a complex step. For now, we'll just create it
                    // and let the user generate a prompt.
                } catch (err) {
                    console.error("Failed to create cover letter", err);
                }
            } else {
                // Fetch existing (not implemented in client, but would be)
                // setCoverLetter(fetchedCoverLetter);
            }
        };
        setupCoverLetter();
    }, [application]);

    const handleGeneratePrompt = async () => {
        setIsLoading(true);
        try {
            // This is a placeholder. In a real UI, you'd build
            // Ideas and Paragraphs first by calling:
            // await addCoverLetterIdea(coverLetter.id, "My Key Idea", [mapping.pairs[0].id]);
            // await addCoverLetterParagraph(coverLetter.id, [newIdea.id], "Body");
            
            const res = await generateCoverLetterPrompt(mapping.id);
            setClPromptJson(JSON.stringify(res.data, null, 2));
            setIsModalOpen(true);
        } catch (err) {
            alert("Failed to generate prompt. (Ensure CoverLetter, Ideas, and Paragraphs exist)");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div>
            <h4 className="h5">Step 3: Build Your Cover Letter</h4>
            <p className="text-muted">
                Group your mapped points into key ideas, then arrange them into paragraphs.
            </p>
            
            <div className="border p-3 rounded bg-light" style={{ minHeight: '300px' }}>
                <h6 className="h5">Phase 1: Group Mappings into Ideas</h6>
                <p className="fst-italic">(UI for creating 'Idea' objects would go here)</p>
                
                <h6 className="h5 mt-4">Phase 2: Arrange Ideas into Paragraphs</h6>
                <p className="fst-italic">(UI for creating 'Paragraph' objects would go here)</p>
            </div>

            <button 
                className="btn btn-info mt-3" 
                onClick={handleGeneratePrompt}
                disabled={isLoading}
            >
                {isLoading ? "Generating..." : "Generate Cover Letter Prompt"}
            </button>

            <PromptModal
                isOpen={isModalOpen}
                jsonString={clPromptJson}
                onClose={() => setIsModalOpen(false)}
            />
            
            <div className="d-flex justify-content-between mt-4">
                <button className="btn btn-secondary" onClick={onPrev}>
                    &lt; Back: Generate CV
                </button>
                <button className="btn btn-primary" onClick={onNext}>
                    Next: Submit & Track &gt;
                </button>
            </div>
        </div>
    );
};

export default Step3_BuildCoverLetter;