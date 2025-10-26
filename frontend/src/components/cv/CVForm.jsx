// frontend/src/components/CVForm.jsx
import React, { useState } from 'react';

const CVForm = ({ onCreate }) => {
    const [name, setName] = useState('');
    const [summary, setSummary] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (name.trim()) {
            // Call the handler function passed from App.jsx
            onCreate(name, summary);
            setName('');
            setSummary('');
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '15px', borderRadius: '5px', textAlign: 'left', backgroundColor: '#f0f0f0', color: '#333' }}>
            <h3 style={{ marginTop: 0 }}>Create New Base CV</h3>
            <div style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', fontWeight: 'bold' }}>
                    CV Name (Required):
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                        style={{ marginLeft: '10px', padding: '5px', width: '90%', boxSizing: 'border-box' }}
                    />
                </label>
            </div>
            <div style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', fontWeight: 'bold' }}>
                    Summary:
                    <textarea
                        value={summary}
                        onChange={(e) => setSummary(e.target.value)}
                        style={{ display: 'block', width: '98%', padding: '5px', minHeight: '80px', marginTop: '5px' }}
                    />
                </label>
            </div>
            <button type="submit" style={{ backgroundColor: '#007bff', color: 'white', border: 'none', padding: '10px 20px', cursor: 'pointer', borderRadius: '4px' }}>
                Create CV Record
            </button>
        </form>
    );
};

export default CVForm;