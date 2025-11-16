// frontend/src/components/Navbar.jsx
import React, { useState, useEffect, useRef } from 'react'; // Make sure useRef is imported
import { NavLink, Link } from 'react-router-dom';
import { LayoutDashboard, FileText, Briefcase, Target, Sun, Moon, Settings } from 'lucide-react';
import logoLight from '../assets/logo_light.svg';
import logoDark from '../assets/logo_dark.svg';

const Navbar = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  // --- 1. CREATE REFS ---
  const togglerRef = useRef(null);
  const collapseRef = useRef(null);

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

  // --- 2. CREATE A CLICK HANDLER ---
  const handleNavClick = () => {
    // Check if the collapse menu is currently open (has 'show' class)
    if (collapseRef.current && collapseRef.current.classList.contains('show')) {
      // If it is, programmatically click the toggler button to close it
      togglerRef.current.click();
    }
  };

  const navItems = [
    { to: '/', label: 'Overview', icon: LayoutDashboard },
    { to: '/applications', label: 'Applications', icon: Briefcase },
    { to: '/cv', label: 'CV Library', icon: FileText },
    { to: '/goals', label: 'Goals', icon: Target },
  ];

  return (
    <nav className="navbar navbar-expand-lg fixed-top navbar-glass border-bottom">
      <div className="container-xxl">
        
        {/* Brand */}
        <Link to="/" className="d-flex align-items-center text-decoration-none">
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
        </Link>

        {/* Mobile Toggle */}
        <button 
          // --- 3. ATTACH THE REF ---
          ref={togglerRef}
          className="navbar-toggler" 
          type="button" 
          data-bs-toggle="collapse" 
          data-bs-target="#navbarContent"
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        {/* Links */}
        <div 
          // --- 3. ATTACH THE REF ---
          ref={collapseRef}
          className="collapse navbar-collapse justify-content-center" 
          id="navbarContent"
        >
            <ul className="navbar-nav mb-2 mb-lg-0 gap-2">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    return (
                        <li className="nav-item" key={item.to}>
                            <NavLink
                                to={item.to}
                                className={({ isActive }) => 
                                    `nav-link d-flex align-items-center gap-2 px-3 rounded-3 ${isActive ? 'active bg-primary-subtle text-primary fw-semibold' : ''}`
                                }
                                style={{ border: 'none', background: 'transparent' }}
                                // --- 4. REMOVE data-bs-* and ADD onClick ---
                                onClick={handleNavClick}
                            >
                                <Icon size={18} />
                                {item.label}
                            </NavLink>
                        </li>
                    );
                })}
            </ul>
        </div>

        {/* Right Actions */}
        <div className="d-none d-lg-flex align-items.center gap-2">
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