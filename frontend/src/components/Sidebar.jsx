import React, { useEffect, useState } from 'react';
import { 
  LayoutDashboard, 
  Briefcase, 
  FileText, 
  Target, 
  Settings, 
  Moon, 
  Sun, 
  ChevronRight 
} from 'lucide-react';
import logoLight from '../assets/logo_light.svg';
import logoDark from '../assets/logo_dark.svg';

const Sidebar = ({ activeView, setActiveView }) => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Theme Toggle Logic
  useEffect(() => {
    const checkTheme = () => {
        const isDark = document.documentElement.classList.contains('dark');
        setIsDarkMode(isDark);
    };
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  const toggleTheme = () => {
    if (isDarkMode) {
      document.documentElement.classList.remove('dark');
    } else {
      document.documentElement.classList.add('dark');
    }
  };

  const navItems = [
    { key: 'Dashboard', label: 'Command Center', icon: LayoutDashboard },
    { key: 'Application_Tracker', label: 'Active Jobs', icon: Briefcase },
    { key: 'CV_Manager', label: 'Master CVs', icon: FileText },
    { key: 'Goal_Tracker', label: 'Goals', icon: Target },
  ];

  return (
    <div className="h-full flex flex-col justify-between py-6 px-4">
      {/* 1. Header / Brand */}
      <div className="mb-10 px-2">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
            R
          </div>
          <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
            RoleCraft
          </span>
        </div>
        <p className="text-xs text-slate-500 font-medium ml-11">Career Operating System</p>
      </div>

      {/* 2. Main Navigation */}
      <nav className="space-y-1 flex-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.key;
          return (
            <button
              key={item.key}
              onClick={() => setActiveView(item.key)}
              className={`
                w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group
                ${isActive 
                  ? 'bg-slate-100 dark:bg-slate-800 text-blue-600 dark:text-blue-400' 
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'}
              `}
            >
              <div className="flex items-center gap-3">
                <Icon size={18} className={isActive ? 'text-blue-600 dark:text-blue-400' : 'text-slate-400 group-hover:text-slate-600'} />
                {item.label}
              </div>
              {isActive && <div className="w-1.5 h-1.5 rounded-full bg-blue-600"></div>}
            </button>
          );
        })}
      </nav>

      {/* 3. Footer / Settings */}
      <div className="pt-6 border-t border-slate-100 dark:border-slate-800 space-y-2">
        <button 
          onClick={toggleTheme}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
        >
          {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          {isDarkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
        
        <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
          <Settings size={18} />
          Settings
        </button>
      </div>
    </div>
  );
};

export default Sidebar;