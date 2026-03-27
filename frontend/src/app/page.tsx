import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-slate-50 dark:bg-slate-950">
      <header className="px-6 h-16 flex items-center justify-between border-b bg-white dark:bg-slate-900">
        <div className="flex items-center gap-2">
          {/* A simple logo placeholder */}
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
            <span className="text-white font-bold text-lg">W</span>
          </div>
          <span className="font-semibold text-lg">WellbeingMonitor</span>
        </div>
        <nav>
          <Link href="/login">
            <Button>Sign In</Button>
          </Link>
        </nav>
      </header>
      
      <main className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-4xl mx-auto">
        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight mb-6">
          Empowering Workplace <span className="text-blue-600">Wellbeing</span> Through AI
        </h1>
        <p className="text-xl text-slate-600 dark:text-slate-400 mb-10 max-w-2xl">
          Track, understand, and improve emotional health across your organization with advanced sentiment analysis and actionable insights.
        </p>
        <div className="flex flex-col sm:flex-row gap-4">
          <Link href="/login">
             <Button size="lg" className="w-full sm:w-auto text-lg px-8">Get Started</Button>
          </Link>
          <Button size="lg" variant="outline" className="w-full sm:w-auto text-lg px-8">Learn More</Button>
        </div>
      </main>
      
      <footer className="py-6 text-center text-sm text-slate-500 border-t bg-white dark:bg-slate-900">
        &copy; {new Date().getFullYear()} WellbeingMonitor. All rights reserved.
      </footer>
    </div>
  );
}
