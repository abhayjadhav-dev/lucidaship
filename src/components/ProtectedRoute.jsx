import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '@clerk/clerk-react';

export default function ProtectedRoute({ children }) {
  const { isLoaded, isSignedIn } = useAuth();

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="font-mono text-[0.7rem] tracking-[0.2em] uppercase text-accent">Loading session...</div>
      </div>
    );
  }

  if (!isSignedIn) {
    return <Navigate to="/login" replace />;
  }

  return children;
}
