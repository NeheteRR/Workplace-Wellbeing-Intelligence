"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";

interface User {
  employee_id: string;
  role: string;
  org_id: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  login: (userData: User, token: string) => void;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // Check for token and user data in localStorage on mount
    const storedUser = localStorage.getItem("user");
    const token = localStorage.getItem("token");

    if (storedUser && token) {
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    // Protect routes
    if (!loading) {
      const isPublic = pathname === "/" || pathname === "/login";
      if (!user && !isPublic) {
        router.push("/login");
      } else if (user) {
        // Redirect if trying to access public routes while logged in
        if (pathname === "/login" || pathname === "/") {
          if (user.role === "hr") {
             router.push("/hr/dashboard");
          } else {
             router.push("/employee/dashboard");
          }
        }
        
        // Prevent employee from accessing hr routes and vice-versa
        if (pathname?.startsWith("/hr") && user.role !== "hr") {
            router.push("/employee/dashboard");
        } else if (pathname?.startsWith("/employee") && user.role === "hr") {
            router.push("/hr/dashboard");
        }
      }
    }
  }, [user, loading, pathname, router]);

  const login = (userData: User, token: string) => {
    localStorage.setItem("user", JSON.stringify(userData));
    localStorage.setItem("token", token);
    localStorage.setItem("org_id", userData.org_id);
    setUser(userData);
    
    if (userData.role === "hr") {
      router.push("/hr/dashboard");
    } else {
      router.push("/employee/dashboard");
    }
  };

  const logout = () => {
    localStorage.removeItem("user");
    localStorage.removeItem("token");
    localStorage.removeItem("org_id");
    router.push("/login");
    // Delay setUser to null slightly to allow UI transitions and menu close events to finish
    // preventing abrupt component unmount exceptions from the routing change.
    setTimeout(() => {
      setUser(null);
    }, 100);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
