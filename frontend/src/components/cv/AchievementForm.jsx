import React, { useState } from 'react';

const AchievementForm = ({ onSubmit, cvId }) => {
    const [text, setText] = useState('');
    
    const handleSubmit = (e) => {
        e.preventDefault();
        if (text.trim()) {
            // Achievement only requires 'text' and 'context' (we default context to 'Global' here)
            onSubmit(cvId, { text, context: 'Global' }, 'Achievement'); 
            setText('');
        }
    };
    return (
        <form onSubmit={handleSubmit} style={{ margin: '10px 0', padding: '15px', border: '1px solid #6c757d', borderRadius: '5px', backgroundColor: '#e9ecef', textAlign: 'left' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#6c757d' }}>+ Add Master Achievement</h4>
            <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Achievement text (e.g., Optimized X by Y%)" required style={{ width: '95%', padding: '8px', minHeight: '60px', marginBottom: '10px' }} />
            <button type="submit" style={{ backgroundColor: '#6c757d', color: 'white' }}>Create Achievement</button>
        </form>
    );
};

export default AchievementForm;