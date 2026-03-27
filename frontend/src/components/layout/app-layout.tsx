"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/components/providers/auth-provider";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { 
  BarChart3, 
  LayoutDashboard, 
  MessageSquareHeart, 
  Activity, 
  Users, 
  Settings, 
  LogOut,
  Menu,
  X
} from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuGroup, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export function AppLayout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  if (!user) return null; // Or a loading spinner

  const isHr = user.role === "hr";

  const navItems = isHr ? [
    { href: "/hr/dashboard", label: "Overview", icon: LayoutDashboard },
    { href: "/hr/emotions", label: "Emotions", icon: MessageSquareHeart },
    { href: "/hr/departments", label: "Departments", icon: Users },
    { href: "/hr/trends", label: "Trends", icon: BarChart3 },
    { href: "/hr/risk", label: "Risk Signals", icon: Activity },
  ] : [
    { href: "/employee/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { href: "/employee/checkin", label: "Daily Check-in", icon: MessageSquareHeart },
    { href: "/employee/wellbeing", label: "Wellbeing History", icon: Activity },
    { href: "/employee/emotions", label: "Emotion Logs", icon: BarChart3 },
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex text-slate-900 dark:text-slate-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-slate-950 border-r transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static lg:block ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="h-16 flex items-center px-6 border-b">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center mr-2">
            <span className="text-white font-bold text-lg">W</span>
          </div>
          <span className="font-semibold text-lg whitespace-nowrap overflow-hidden text-ellipsis">Wellbeing System</span>
          <button className="ml-auto lg:hidden" onClick={() => setSidebarOpen(false)}>
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link key={item.href} href={item.href}>
                <Button 
                   variant={isActive ? "secondary" : "ghost"} 
                   className={`w-full justify-start ${isActive ? 'bg-slate-100 dark:bg-slate-800' : ''}`}
                   onClick={() => setSidebarOpen(false)}
                >
                  <item.icon className="mr-2 h-4 w-4" />
                  {item.label}
                </Button>
              </Link>
            );
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 bg-white dark:bg-slate-950 border-b flex items-center justify-between px-4 sm:px-6">
          <div className="flex items-center">
            <button className="mr-4 lg:hidden" onClick={() => setSidebarOpen(true)}>
              <Menu className="w-5 h-5" />
            </button>
            <h2 className="font-semibold text-lg hidden sm:block">
              {isHr ? "HR Admin Portal" : "Employee Portal"}
            </h2>
          </div>
          
          <div className="flex items-center gap-4">
            <DropdownMenu>
              <DropdownMenuTrigger className="relative h-8 w-8 rounded-full outline-none focus:ring-2 focus:ring-slate-400">
                  <Avatar className="h-8 w-8 bg-blue-100 text-blue-700">
                    <AvatarFallback>{user.name.charAt(0)}</AvatarFallback>
                  </Avatar>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end">
                <DropdownMenuGroup>
                  <DropdownMenuLabel className="font-normal">
                    <div className="flex flex-col space-y-1">
                      <p className="text-sm font-medium leading-none">{user.name}</p>
                      <p className="text-xs leading-none text-muted-foreground">
                        {user.employee_id} • {user.role === 'hr' ? 'HR' : 'Employee'}
                      </p>
                    </div>
                  </DropdownMenuLabel>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="cursor-pointer" onClick={logout}>
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        <main className="flex-1 overflow-auto p-4 sm:p-6 lg:p-8">
          <div className="mx-auto max-w-6xl">
             {children}
          </div>
        </main>
      </div>
    </div>
  );
}
