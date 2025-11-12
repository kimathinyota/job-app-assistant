import React, { useState, useEffect } from 'react';
import { LayoutDashboard, FileText, Briefcase, Target, Sun, Moon, Settings } from 'lucide-react';
import logoLight from '../assets/logo_light.svg';
import logoDark from '../assets/logo_dark.svg';

const Navbar = ({ activeView, setActiveView }) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 10);
    window.addEventListener('scroll', handleScroll);
    
    const checkTheme = () => setIsDarkMode(document.documentElement.classList.contains('dark'));
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    
    return () => {
        window.removeEventListener('scroll', handleScroll);
        observer.disconnect();
    };
  }, []);

  const toggleTheme = () => {
    document.documentElement.classList.toggle('dark');
    setIsDarkMode(!isDarkMode);
  };

  const navItems = [
    { key: 'Dashboard', label: 'Overview', icon: LayoutDashboard },
    { key: 'Application_Tracker', label: 'Applications', icon: Briefcase },
    { key: 'CV_Manager', label: 'CV Library', icon: FileText },
    { key: 'Goal_Tracker', label: 'Goals', icon: Target },
  ];

  return (
    <header 
      className={`sticky top-0 z-50 w-full transition-all duration-200 ${
        isScrolled 
          ? 'bg-white/80 dark:bg-slate-950/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800' 
          : 'bg-transparent border-b border-transparent'
      }`}
      // Fallback style if Tailwind fails to load
      style={{ 
        position: 'sticky', top: 0, zIndex: 1030, 
        backgroundColor: isScrolled ? 'rgba(255,255,255,0.9)' : 'transparent',
        borderBottom: isScrolled ? '1px solid #e2e8f0' : 'none'
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between" style={{ display: 'flex', height: '64px', alignItems: 'center', justifyContent: 'space-between' }}>
          
          {/* 1. Brand Identity */}
          <div 
            className="flex items-center gap-3 cursor-pointer group"
            onClick={() => setActiveView('Dashboard')}
            style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
          >
            {/* FORCE LOGO SIZE HERE */}
            <div className="relative w-8 h-8 flex items-center justify-center" style={{ width: '32px', height: '32px', display: 'flex', alignItems: 'center' }}>
               <img 
                  src={isDarkMode ? logoDark : logoLight} 
                  alt="RoleCraft" 
                  className="w-full h-full object-contain"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  onError={(e) => { e.target.style.display = 'none'; }} 
               />
            </div>
            
            <div className="flex flex-col" style={{ display: 'flex', flexDirection: 'column' }}>
              <span className="text-lg font-bold tracking-tight text-slate-900 dark:text-white leading-none" style={{ fontSize: '18px', fontWeight: 'bold', lineHeight: 1 }}>
                RoleCraft
              </span>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400" style={{ fontSize: '10px', textTransform: 'uppercase', color: '#64748b' }}>
                Career OS
              </span>
            </div>
          </div>

          {/* 2. Center Navigation */}
          <nav className="hidden md:flex items-center gap-1" style={{ display: 'flex', gap: '4px' }}>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeView === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => setActiveView(item.key)}
                  className={`
                    flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200
                    ${isActive 
                      ? 'bg-slate-100 dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm ring-1 ring-slate-200 dark:ring-slate-700' 
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800'}
                  `}
                >
                  <Icon size={16} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* 3. Right Actions */}
          <div className="flex items-center gap-2" style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={toggleTheme}
              className="p-2 text-slate-500 hover:text-slate-700 rounded-full transition-colors"
            >
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button className="p-2 text-slate-500 hover:text-slate-700 rounded-full transition-colors">
              <Settings size={18} />
            </button>
          </div>

        </div>
      </div>
    </header>
  );
};

export default Navbar;