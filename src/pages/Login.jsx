import React, { useEffect } from 'react';
import { SignIn, useAuth } from '@clerk/clerk-react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const { isLoaded, isSignedIn } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isLoaded && isSignedIn) {
      navigate('/dashboard', { replace: true });
    }
  }, [isLoaded, isSignedIn, navigate]);

  return (
    <div className="min-h-screen w-full flex items-center justify-center relative bg-black p-4">
      <div className="bg-glow"></div>
      <div className="relative z-10">
        <SignIn
          signUpUrl="/register"
          forceRedirectUrl="/dashboard"
          fallbackRedirectUrl="/dashboard"
          routing="path"
          path="/login"
        />
      </div>
    </div>
  );
}
