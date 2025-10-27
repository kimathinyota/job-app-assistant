import React from 'react';

const NestedList = ({ cvId, items, title, listName, onRefresh, onDelete }) => {
    
    if (!items || items.length === 0) {
        return (
            <div style={{ padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
                <h4 style={{ margin: 0, color: '#007bff' }}>{title} (0)</h4>
                <p style={{ fontSize: '0.8em', color: '#888' }}>No {listName} added yet.</p>
            </div>
        );
    }
    
    return (
        <div style={{ padding: '10px', border: '1px solid #ccc', borderRadius: '5px', backgroundColor: '#f9f9f9' }}>
            <h4 style={{ margin: '10px 0', color: '#007bff' }}>{title} ({items.length})</h4>
            {items.map(item => (
                <div key={item.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px dotted #ccc', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.9em', color: '#333' }}>
                        {/* Displays the most relevant property: title, name, or text */}
                        {item.title || item.name || item.text}
                    </span>
                    <div>
                        {/* Edit Button Placeholder */}
                        <button style={{ fontSize: '0.7em', padding: '3px 8px', marginRight: '5px', backgroundColor: '#17a2b8', color: 'white' }}>
                            Edit
                        </button>
                        {/* Delete Button - Calls the handler passed down from CVManagerPage */}
                        <button 
                            onClick={() => onDelete(cvId, item.id, listName)} 
                            style={{ fontSize: '0.7em', padding: '3px 8px', backgroundColor: '#dc3545', color: 'white' }}
                        >
                            Delete
                        </button>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default NestedList;