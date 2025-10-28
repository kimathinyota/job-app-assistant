import React from 'react';

const NestedList = ({ cvId, items, title, listName, onDelete, onRefresh, onEdit }) => (
    <div style={{ padding: '10px' }}>
        <h3>{title}</h3>
        {items.length === 0 ? (
            <p>No {listName} added yet.</p>
        ) : (
            <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
                {items.map((item) => (
                    <li key={item.id} style={{ border: '1px solid #ddd', marginBottom: '10px', padding: '10px', borderRadius: '5px' }}>
                        <strong>{item.title || item.name}</strong>
                        {item.company && <p>{item.company}</p>}
                        {item.description && <p>{item.description}</p>}
                        <div style={{ marginTop: '10px' }}>
                            {onEdit && (
                                <button
                                    onClick={() => onEdit(item)}
                                    style={{ marginRight: '10px', backgroundColor: '#ffc107', color: 'black', padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer' }}
                                >
                                    Edit
                                </button>
                            )}
                            <button
                                onClick={() => onDelete(cvId, item.id, listName)}
                                style={{ backgroundColor: '#dc3545', color: 'white', padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer' }}
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
