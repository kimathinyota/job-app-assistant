import React, { useState, useEffect } from 'react';
import { LayoutDashboard, FileText, Briefcase, Target, Sun, Moon, Settings } from 'lucide-react';
import logoLight from '../assets/logo_light.svg';
import logoDark from '../assets/logo_dark.svg';

const Navbar = ({ activeView, setActiveView }) => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Handle Theme Persistence
  useEffect(() => {
    const storedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (storedTheme === 'dark' || (!storedTheme && prefersDark)) {
      setTheme(true);
    }
  }, []);

  const setTheme = (isDark) => {
    document.documentElement.setAttribute('data-bs-theme', isDark ? 'dark' : 'light');
    setIsDarkMode(isDark);
  };

  const toggleTheme = () => {
    const newMode = !isDarkMode;
    setTheme(newMode);
    localStorage.setItem('theme', newMode ? 'dark' : 'light');
  };

  const navItems = [
    { key: 'Dashboard', label: 'Overview', icon: LayoutDashboard },
    { key: 'Application_Tracker', label: 'Applications', icon: Briefcase },
    { key: 'CV_Manager', label: 'CV Library', icon: FileText },
    { key: 'Goal_Tracker', label: 'Goals', icon: Target },
  ];

  return (
    // FIXED-TOP: This class pins the navbar to the top of the viewport
    <nav className="navbar navbar-expand-lg fixed-top navbar-glass border-bottom">
      <div className="container-xxl">
        
        {/* Brand */}
        <div 
            className="d-flex align-items-center cursor-pointer" 
            onClick={() => setActiveView('Dashboard')}
            style={{ cursor: 'pointer' }}
        >
            <img 
                src={isDarkMode ? logoDark : logoLight} 
                alt="RoleCraft Logo" 
                width="32" 
                height="32"
                className="me-2"
            />
            <div className="d-flex flex-column">
                <span className="fw-bold lh-1" style={{ color: 'var(--brand-text)' }}>
                    RoleCraft
                </span>
                <span className="text-uppercase fw-bold" style={{ fontSize: '0.65rem', color: 'var(--brand-text-muted)' }}>
                    Career OS
                </span>
            </div>
        </div>

        {/* Mobile Toggle */}
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarContent">
          <span className="navbar-toggler-icon"></span>
        </button>

        {/* Links */}
        <div className="collapse navbar-collapse justify-content-center" id="navbarContent">
            <ul className="navbar-nav mb-2 mb-lg-0 gap-2">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = activeView === item.key;
                    return (
                        <li className="nav-item" key={item.key}>
                            <button
                                onClick={() => setActiveView(item.key)}
                                className={`nav-link d-flex align-items-center gap-2 px-3 rounded-3 ${isActive ? 'active bg-primary-subtle text-primary fw-semibold' : ''}`}
                                style={{ border: 'none', background: isActive ? '' : 'transparent' }}
                            >
                                <Icon size={18} />
                                {item.label}
                            </button>
                        </li>
                    );
                })}
            </ul>
        </div>

        {/* Right Actions */}
        <div className="d-none d-lg-flex align-items-center gap-2">
            <button onClick={toggleTheme} className="btn btn-link nav-link p-2">
                {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <div className="vr mx-2"></div>
            <button className="btn btn-link nav-link p-2">
                <Settings size={20} />
            </button>
        </div>

      </div>
    </nav>
  );
};

export default Navbar;