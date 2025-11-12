// frontend/src/components/NavMenu.jsx
import React, { useEffect, useState } from 'react';
import logoLight from '../assets/logo_light.svg';
import logoDark from '../assets/logo_dark.svg';

const NavMenu = ({ activeView, setActiveView }) => {
    // State to track theme for logo switching
    const [isDarkMode, setIsDarkMode] = useState(false);

    // Logic to check for .dark class on mount and changes
    useEffect(() => {
        const checkTheme = () => {
            const isDark = document.documentElement.classList.contains('dark');
            setIsDarkMode(isDark);
        };

        // Initial check
        checkTheme();

        // Optional: Listen for class changes if you have a dynamic toggle elsewhere
        const observer = new MutationObserver(checkTheme);
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

        return () => observer.disconnect();
    }, []);

    const navItems = [
        { key: 'Dashboard', label: 'Overview' },
        { key: 'CV_Manager', label: 'Career Profile' },
        { key: 'Application_Tracker', label: 'Applications' },
        { key: 'Goal_Tracker', label: 'Goals' },
    ];

    const getButtonStyle = (key) => {
        const isActive = activeView === key;
        return {
            padding: '0.5rem 1rem',
            margin: '0 0.25rem',
            cursor: 'pointer',
            backgroundColor: isActive ? 'var(--brand-accent)' : 'transparent',
            color: isActive ? '#fff' : 'var(--text-muted)',
            border: 'none',
            borderRadius: '0.375rem', // Tailwind rounded-md
            fontWeight: 500,
            fontSize: '0.9rem',
            transition: 'all 0.2s ease',
        };
    };

    return (
        <nav className="navbar-glass mb-4">
            <div className="container-fluid" style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px' }}>
                
                {/* Brand Identity */}
                <div 
                    className="d-flex align-items-center" 
                    onClick={() => setActiveView('Dashboard')}
                    style={{ cursor: 'pointer' }}
                >
                    <img 
                        src={isDarkMode ? logoDark : logoLight} 
                        alt="RoleCraft Logo" 
                        style={{ height: '32px', marginRight: '12px' }} 
                        onError={(e) => { e.target.style.display = 'none'; }} // Fallback if assets missing
                    />
                    <div className="d-flex flex-column text-start">
                        <span style={{ fontSize: '1.25rem', fontWeight: '700', lineHeight: '1.1', letterSpacing: '-0.03em', color: 'var(--text-main)' }}>
                            RoleCraft
                        </span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: '500' }}>
                            Career OS
                        </span>
                    </div>
                </div>

                {/* Navigation Links */}
                <div className="d-flex align-items-center">
                    {navItems.map(item => (
                        <button
                            key={item.key}
                            style={getButtonStyle(item.key)}
                            onClick={() => setActiveView(item.key)}
                            className="nav-btn"
                        >
                            {item.label}
                        </button>
                    ))}
                </div>
            </div>
        </nav>
    );
};

export default NavMenu;