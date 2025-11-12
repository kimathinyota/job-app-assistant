import React from 'react';
import Navbar from './Navbar';

const Layout = ({ children, activeView, setActiveView }) => {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 font-sans text-slate-900 dark:text-slate-50 selection:bg-blue-100 dark:selection:bg-blue-900">
      
      {/* Top Navigation */}
      <Navbar activeView={activeView} setActiveView={setActiveView} />

      {/* Main Content Area */}
      <main className="animate-in fade-in duration-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </div>
      </main>

    </div>
  );
};

export default Layout;