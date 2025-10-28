import React from 'react';

// Added onEdit prop
const NestedList = ({ cvId, items, title, listName, onDelete, onRefresh, onEdit }) => (
    <div style={{ padding: '10px' }}>
        <h3>{title} ({items?.length || 0})</h3>
        {!items || items.length === 0 ? (
            <p style={{ color: '#666', fontStyle: 'italic' }}>No {listName} added yet.</p>
        ) : (
            <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
                {items.map((item) => (
                    <li key={item.id} style={{ border: '1px solid #ddd', marginBottom: '10px', padding: '10px', borderRadius: '5px', backgroundColor: '#fff' }}>
                        {/* Display relevant info */}
                        <strong>{item.title || item.name || item.text}</strong>
                        {item.company && <p style={{ margin: '5px 0', fontSize: '0.9em', color: '#555' }}>{item.company}</p>}
                        {item.description && <p style={{ margin: '5px 0', fontSize: '0.9em', color: '#555' }}>{item.description}</p>}
                        {/* You could add skill/achievement tags here later */}

                        <div style={{ marginTop: '10px' }}>
                            {/* --- EDIT BUTTON --- */}
                            {onEdit && ( // Only show if onEdit handler is provided
                                <button
                                    onClick={() => onEdit(item)} // Pass the full item object
                                    style={{
                                        marginRight: '10px',
                                        backgroundColor: '#ffc107', color: '#333',
                                        padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '0.8em'
                                    }}
                                >
                                    Edit
                                </button>
                            )}
                            {/* --- DELETE BUTTON --- */}
                            <button
                                onClick={() => onDelete(cvId, item.id, listName)}
                                style={{
                                    backgroundColor: '#dc3545', color: 'white',
                                    padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '0.8em'
                                }}
                            >
                                Delete
                            </button>
                        </div>
                    </li>
                ))}
            </ul>
        )}
    </div>
);

export default NestedList;