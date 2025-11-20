import React, { useState, useEffect, useRef } from 'react'; 
import { NavLink, Link } from 'react-router-dom';
import { LayoutDashboard, FileText, Briefcase, Target, Sun, Moon, Settings } from 'lucide-react';
import logoLight from '../assets/logo_light.svg';
import logoDark from '../assets/logo_dark.svg';

const Navbar = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  const togglerRef = useRef(null);
  const collapseRef = useRef(null);

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

  const handleNavClick = () => {
    if (collapseRef.current && collapseRef.current.classList.contains('show')) {
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
                alt="RoleCase Logo" 
                width="32" 
                height="32"
                className="me-2"
            />
            <div className="d-flex flex-column">
                <span className="fw-bold lh-1" style={{ color: 'var(--brand-text)' }}>
                    RoleCase
                </span>
                <span className="text-uppercase fw-bold" style={{ fontSize: '0.65rem', color: 'var(--brand-text-muted)' }}>
                    Career OS
                </span>
            </div>
        </Link>

        {/* Mobile Toggle */}
        <button 
          ref={togglerRef}
          className="navbar-toggler" 
          type="button" 
          data-bs-toggle="collapse" 
          data-bs-target="#navbarContent"
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        {/* Links & Actions Container */}
        <div 
          ref={collapseRef}
          className="collapse navbar-collapse" 
          id="navbarContent"
        >
            {/* Main Navigation Links */}
            <ul className="navbar-nav mx-auto mb-2 mb-lg-0 gap-2 p-2 p-lg-0">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    return (
                        <li className="nav-item" key={item.to}>
                            <NavLink
                                to={item.to}
                                className={({ isActive }) => 
                                    `nav-link d-flex align-items-center gap-2 px-3 rounded-3 ${isActive ? 'active bg-primary-subtle text-primary fw-semibold' : ''}`
                                }
                                onClick={handleNavClick}
                            >
                                <Icon size={18} />
                                {item.label}
                            </NavLink>
                        </li>
                    );
                })}
                
                {/* Mobile-Only Actions (Divider + Buttons) */}
                <li className="nav-item d-lg-none mt-2 pt-2 border-top">
                    <div className="d-flex align-items-center justify-content-between px-3">
                        <span className="text-muted small fw-bold text-uppercase">Settings</span>
                        <div className="d-flex gap-3">
                            <button onClick={toggleTheme} className="btn btn-sm btn-light border d-flex align-items-center gap-2">
                                {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
                                {isDarkMode ? 'Light Mode' : 'Dark Mode'}
                            </button>
                            <button className="btn btn-sm btn-light border">
                                <Settings size={16} />
                            </button>
                        </div>
                    </div>
                </li>
            </ul>

            {/* Desktop-Only Right Actions */}
            <div className="d-none d-lg-flex align-items-center gap-2">
                <button onClick={toggleTheme} className="btn btn-link nav-link p-2" title="Toggle Theme">
                    {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
                </button>
                <div className="vr mx-2"></div>
                <button className="btn btn-link nav-link p-2" title="Settings">
                    <Settings size={20} />
                </button>
            </div>

        </div>
      </div>
    </nav>
  );
};

export default Navbar;