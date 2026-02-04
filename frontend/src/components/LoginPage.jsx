// frontend/src/components/LoginPage.jsx
import React from 'react';
import logo from '../assets/logo_dark.svg'; 

export default function LoginPage() {
  
  const handleLogin = () => {
    // Redirect the browser to the Backend's Google OAuth endpoint
    window.location.href = "http://localhost:8000/auth/login/google";
  };

  return (
    <div className="min-vh-100 w-100 d-flex flex-column align-items-center justify-content-center bg-light">
      <div className="w-100 p-4" style={{ maxWidth: '28rem' }}>
        <div className="card shadow-lg border-0">
          <div className="card-header bg-white border-0 text-center pb-2 pt-4">
            <div className="mx-auto mb-4 d-flex align-items-center justify-content-center bg-dark rounded-3" style={{ width: '4rem', height: '4rem' }}>
              <img src={logo} alt="Logo" style={{ width: '2.5rem', height: '2.5rem' }} />
            </div>
            <h2 className="card-title h4 fw-bold text-dark mb-0">
              RoleCase
            </h2>
            <p className="text-muted small mt-2 mb-0">
              Sign in to manage your CVs and Applications
            </p>
          </div>
          
          <div className="card-body pt-4 pb-5 px-5">
            <button 
              onClick={handleLogin}
              className="btn btn-outline-secondary w-100 d-flex align-items-center justify-content-center gap-3 py-2 fw-medium shadow-sm"
              style={{ backgroundColor: 'white', borderColor: '#dee2e6' }}
              onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#f8f9fa'}
              onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'white'}
            >
              <svg className="bi" width="20" height="20" viewBox="0 0 24 24">
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  fill="#EA4335"
                />
              </svg>
              Continue with Google
            </button>
            
            <div className="mt-4 text-center" style={{ fontSize: '0.75rem', color: '#adb5bd' }}>
              Secured by Google OAuth • No passwords required
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}