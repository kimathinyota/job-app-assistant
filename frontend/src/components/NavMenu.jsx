// frontend/src/components/NavMenu.jsx
import React from 'react';

const NavMenu = ({ activeView, setActiveView }) => {
    // Defines the key navigation points (matching the App.jsx views object)
    const navItems = [
        { key: 'Dashboard', label: 'Dashboard Home' },
        { key: 'CV_Manager', label: 'CV Manager' },
        { key: 'Application_Tracker', label: 'Application Suite' },
        { key: 'Goal_Tracker', label: 'Productivity Goals' },
    ];

    const getButtonStyle = (key) => ({
        padding: '10px 15px',
        margin: '0 8px',
        cursor: 'pointer',
        backgroundColor: activeView === key ? '#007bff' : '#f4f4f4',
        color: activeView === key ? 'white' : '#333',
        border: 'none',
        borderRadius: '5px',
        fontWeight: activeView === key ? 'bold' : 'normal',
        transition: 'background-color 0.3s',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        minWidth: '150px'
    });

    return (
        <nav style={{ marginBottom: '20px', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
            {navItems.map(item => (
                <button
                    key={item.key}
                    style={getButtonStyle(item.key)}
                    onClick={() => setActiveView(item.key)}
                >
                    {item.label}
                </button>
            ))}
        </nav>
    );
};
export default NavMenu;