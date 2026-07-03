'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

interface User {
  username: string;
  ip: string;
  loginTime: number;
  subscription?: {
    status: string;
    expiryDate?: string;
    plan?: string;
  };
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string, ip: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check session on mount
  useEffect(() => {
    const storedUser = localStorage.getItem('auth_user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        // Check if session is still valid (24 hour expiry)
        const sessionAge = Date.now() - parsedUser.loginTime;
        const sessionTimeout = 24 * 60 * 60 * 1000; // 24 hours
        
        if (sessionAge < sessionTimeout) {
          setUser(parsedUser);
        } else {
          localStorage.removeItem('auth_user');
          setError('Session expired. Please login again.');
        }
      } catch {
        localStorage.removeItem('auth_user');
      }
    }
    setIsLoading(false);
  }, []);

  const login = async (username: string, password: string, ip: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, ip }),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.error || 'Login failed');
        throw new Error(data.error || 'Login failed');
      }

      const userData: User = {
        username: data.user.username,
        ip: data.user.ip,
        loginTime: Date.now(),
        subscription: data.user.subscription,
      };

      setUser(userData);
      localStorage.setItem('auth_user', JSON.stringify(userData));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('auth_user');
    setError(null);
  };

  const clearError = () => {
    setError(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        error,
        login,
        logout,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
