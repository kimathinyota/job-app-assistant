// frontend/src/components/cv/AggregatedSkillsDisplay.jsx
import React, { useState, useEffect } from 'react';
import { fetchAggregatedSkills } from '../../api/cvClient';

const AggregatedSkillsDisplay = ({ cvId, listName, itemId }) => {
    const [skills, setSkills] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadSkills = async () => {
            if (!cvId || !listName || !itemId) return;
            setLoading(true);
            try {
                console.log("Fetching aggregated skills for:", cvId, listName, itemId);
                const data = await fetchAggregatedSkills(cvId, listName, itemId);
                setSkills(data || []);
            } catch (error) {
                console.log("Failed to fetch aggregated skills for:", cvId, listName, itemId);
                console.error("Failed to load aggregated skills:", error);
                setSkills([]);
            } finally {
                setLoading(false);
            }
        };
        loadSkills();
    }, [cvId, listName, itemId]); // Re-fetch if props change

    if (loading) {
        return <p style={{ fontSize: '0.85em', color: '#666' }}>Loading skills...</p>;
    }

    if (skills.length === 0) {
        return null; // Don't render anything if no skills are linked
    }

    const skillNames = skills.map(s => s.name).join(', ');

    return (
        <p style={{ margin: '8px 0 0 0', fontSize: '0.85em', fontStyle: 'italic', color: '#666' }}>
            Related Skills: {skillNames}
        </p>
    );
};

export default AggregatedSkillsDisplay;