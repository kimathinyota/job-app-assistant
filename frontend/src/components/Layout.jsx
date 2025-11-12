import React from 'react';
import Navbar from './Navbar';

const Layout = ({ children, activeView, setActiveView }) => {
  return (
    <div className="d-flex flex-column min-vh-100">
      {/* 1. Navigation (Fixed at top) */}
      <Navbar activeView={activeView} setActiveView={setActiveView} />

      {/* 2. Main Content 
          - flex-grow-1: Pushes footer down if you have one
          - container-xxl: Centers content but allows it to get wide (1400px)
          - py-4: Vertical padding inside the content
          - mt-5 pt-5: Top margin/padding to clear the Fixed Navbar
      */}
      <main className="flex-grow-1 mt-5 pt-4">
        <div className="container-xxl py-4">
          {children}
        </div>
      </main>

      {/* Optional Footer */}
      <footer className="text-center py-4 border-top bg-light text-muted">
        <small>© 2025 RoleCraft Inc.</small>
      </footer>
    </div>
  );
};

export default Layout;