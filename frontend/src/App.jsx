// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import { fetchAllCVs, createBaseCV } from './api/client';
import CVList from './components/CVList';
import CVForm from './components/CVForm';
import './App.css'; // Keep original styles

function App() {
    // State to hold the list of CVs fetched from the backend
    const [cvs, setCvs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Function to load data and refresh state
    const loadCvs = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchAllCVs();
            setCvs(data);
        } catch (err) {
            setError('Failed to load CVs. Ensure backend is running and CORS is configured.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    // Load data when the component mounts
    useEffect(() => {
        loadCvs();
    }, []);

    // Function passed to the form to handle creation and then refresh the list
    const handleCreate = async (name, summary) => {
        try {
            await createBaseCV(name, summary);
            // After successful creation, reload the data
            await loadCvs(); 
            alert(`CV '${name}' created successfully!`);
        } catch (err) {
            alert('Error creating CV. Check console for details.');
            console.error(err);
        }
    };

    return (
        <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto', textAlign: 'center' }}>
            <h1 style={{ fontSize: '2.5em' }}>Job Application Assistant</h1>
            
            <section style={{ border: '1px solid #ccc', padding: '20px', borderRadius: '8px', marginBottom: '30px' }}>
                <CVForm onCreate={handleCreate} />
            </section>

            <section style={{ textAlign: 'left' }}>
                <h2 style={{ borderBottom: '2px solid #ccc', paddingBottom: '10px' }}>Your Master CVs ({cvs.length})</h2>
                {loading && <p>Loading CV data...</p>}
                {error && <p style={{ color: 'red', fontWeight: 'bold' }}>Error: {error}</p>}
                {!loading && !error && <CVList cvs={cvs} />}
            </section>
        </div>
    );
}

export default App;