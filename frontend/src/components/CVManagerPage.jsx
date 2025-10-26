import React, { useState, useEffect } from 'react';
import { 
    fetchCVDetails, 
    deleteBaseCV, 
    addExperience, 
    addSkill,
    addAchievement,
    deleteNestedItem,
    linkSkill,
    unlinkSkill // Added unlink
} from '../api/client'; 

// --- Component Imports ---
import CVSelector from './cv/CVList'; // Renamed CVList.jsx to CVSelector for better clarity
import NestedList from './cv/NestedList'; 
import NestedFormStub from './cv/NestedFormStub'; 

// The CV Manager Component: The main hub for CV detailed editing.
const CVManagerPage = ({ cvs, setActiveView, reloadData }) => {
    // Select the first CV by default if the list exists
    const [selectedCVId, setSelectedCVId] = useState(cvs[0]?.id || null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);

    // --- Data Fetching Logic ---
    const fetchAndSetDetails = async (id) => {
        setLoadingDetails(true);
        try {
            const response = await fetchCVDetails(id);
            setDetailedCV(response); 
        } catch (error) {
            console.error("Failed to load CV details:", error);
            setDetailedCV(null);
        } finally {
            setLoadingDetails(false);
        }
    };

    // Effect to handle initial selection and when CVs change
    useEffect(() => {
        if (selectedCVId) {
            fetchAndSetDetails(selectedCVId);
        } else if (cvs.length > 0) {
            setSelectedCVId(cvs[0].id);
        } else {
             setDetailedCV(null);
        }
    }, [selectedCVId, cvs.length]);


    // --- CRUD Handlers ---

    const handleDeleteCV = async (cvId) => {
        if (window.confirm("Are you sure you want to delete this master CV? This is irreversible.")) {
            try {
                await deleteBaseCV(cvId);
                alert("CV deleted successfully!");
                await reloadData(); // Reload master list in App.jsx
                setSelectedCVId(null);
                setDetailedCV(null);
            } catch (error) {
                alert("Failed to delete CV.");
                console.error(error);
            }
        }
    };

    // Generic handler for nested ADD actions (Experience, Skill, etc.)
    const handleAddNestedItem = async (cvId, data, addFunction, itemType) => {
        try {
            // Data structure depends on the specific form fields and the client function signature
            const itemData = itemType === 'Experience' ? { title: data.title, company: data.company, description: data.description } : data;

            await addFunction(cvId, itemData);
            alert(`${itemType} added successfully!`);
            fetchAndSetDetails(cvId); // Reload details of the currently selected CV
        } catch (error) {
            alert(`Failed to add ${itemType}. Check console.`);
            console.error(error);
        }
    };

    // Handler for deleting a nested item
    const handleDeleteNested = async (cvId, itemId, listName) => {
        if (window.confirm(`Delete this item from ${listName}?`)) {
            try {
                await deleteNestedItem(cvId, itemId, listName);
                fetchAndSetDetails(cvId);
            } catch (error) {
                alert(`Error deleting item from ${listName}. Check console.`);
                console.error(error);
            }
        }
    };


    // --- RENDER LOGIC ---

    if (cvs.length === 0) {
        return (
            <div style={{ textAlign: 'center', padding: '50px', border: '2px dashed #007bff', borderRadius: '8px' }}>
                <h3 style={{ color: '#007bff' }}>No CVs Found</h3>
                <p>Please use the form on the Dashboard Home to create your first base CV.</p>
                <button onClick={() => setActiveView('Dashboard')}>Go to Dashboard</button>
            </div>
        );
    }

    return (
        <div style={{ textAlign: 'left' }}>
            <h2>CV Manager & Editor</h2>
            
            {/* 1. CV Selector (Uses the old CVList component) */}
            <CVSelector cvs={cvs} onSelect={setSelectedCVId} selectedCVId={selectedCVId} />

            {/* 2. Detail/Editor Area */}
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px', backgroundColor: '#fff' }}>
                {loadingDetails ? (
                    <p style={{ textAlign: 'center' }}>Loading detailed components...</p>
                ) : detailedCV ? (
                    <div style={{ padding: '10px' }}>
                        <h3 style={{ borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
                            {detailedCV.name} 
                            <button onClick={() => handleDeleteCV(detailedCV.id)} style={{ backgroundColor: '#dc3545', color: 'white', float: 'right', fontSize: '0.8em' }}>
                                Delete CV
                            </button>
                        </h3>
                        
                        {/* --- ADD FORMS --- */}
                        <div style={{ display: 'flex', gap: '20px', marginBottom: '30px', flexWrap: 'wrap' }}>
                             <div style={{ flex: '1 1 45%' }}>
                                <NestedFormStub 
                                    onSubmit={(cvId, data) => handleAddNestedItem(cvId, { title: data.title, company: data.company }, addExperience, 'Experience')} 
                                    itemType="Experience"
                                    cvId={detailedCV.id}
                                />
                             </div>
                             <div style={{ flex: '1 1 45%' }}>
                                <NestedFormStub 
                                    onSubmit={(cvId, data) => handleAddNestedItem(cvId, { name: data.title, category: 'technical' }, addSkill, 'Skill')} 
                                    itemType="Skill"
                                    cvId={detailedCV.id}
                                />
                             </div>
                        </div>
                        
                        {/* --- LISTS --- */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '30px' }}>
                            <div style={{ flex: '1 1 30%' }}>
                                <NestedList 
                                    cvId={detailedCV.id}
                                    items={detailedCV.experiences}
                                    title="Experiences"
                                    listName="experiences"
                                    onRefresh={fetchAndSetDetails}
                                    onDelete={handleDeleteNested}
                                />
                            </div>
                            <div style={{ flex: '1 1 30%' }}>
                                <NestedList 
                                    cvId={detailedCV.id}
                                    items={detailedCV.skills}
                                    title="Master Skills"
                                    listName="skills"
                                    onRefresh={fetchAndSetDetails}
                                    onDelete={handleDeleteNested}
                                />
                            </div>
                            <div style={{ flex: '1 1 30%' }}>
                                <NestedList 
                                    cvId={detailedCV.id}
                                    items={detailedCV.achievements}
                                    title="Achievements"
                                    listName="achievements"
                                    onRefresh={fetchAndSetDetails}
                                    onDelete={handleDeleteNested}
                                />
                            </div>
                        </div>
                    </div>
                ) : (
                    <p style={{ textAlign: 'center', color: '#777' }}>Select a CV above to begin editing its components.</p>
                )}
            </div>
        </div>
    );
};

export default CVManagerPage;
