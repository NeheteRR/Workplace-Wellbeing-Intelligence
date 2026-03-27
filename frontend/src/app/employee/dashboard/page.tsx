"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Activity, Zap, HeartPulse } from "lucide-react";

export default function EmployeeDashboard() {
  const { user } = useAuth();
  const [data, setData] = useState<{ wellbeing_score: number; status: string; emotions: any } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      if (!user) return;
      try {
        const res = await fetchWithAuth(`/employee/today-summary?employee_id=${user.employee_id}`);
        setData(res);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [user]);

  if (loading || !user) {
    return <div className="animate-pulse space-y-4">
       <div className="h-8 bg-slate-200 rounded w-1/4"></div>
       <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="h-32 bg-slate-200 rounded"></div>
          <div className="h-32 bg-slate-200 rounded"></div>
       </div>
    </div>;
  }

  const score = data?.wellbeing_score || 0;
  const status = data?.status || "No Data for Today";
  
  // Format emotions for Recharts
  const emotionData = data?.emotions ? Object.entries(data.emotions).map(([key, value]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1),
    value: Number((Number(value) * 100).toFixed(1)) // convert to percentage
  })) : [];

  const COLORS: Record<string, string> = {
    Joy: "#eab308", // yellow-500
    Sadness: "#3b82f6", // blue-500
    Fear: "#a855f7", // purple-500
    Anger: "#ef4444", // red-500
    Love: "#ec4899", // pink-500
    Neutral: "#94a3b8" // slate-400
  };

  const statusColor = status === "high" ? "text-green-600" : status === "moderate" ? "text-amber-500" : status === "low" ? "text-red-500" : "text-slate-500";
  const statusBg = status === "high" ? "bg-green-100 dark:bg-green-900/30" : status === "moderate" ? "bg-amber-100 dark:bg-amber-900/30" : status === "low" ? "bg-red-100 dark:bg-red-900/30" : "bg-slate-100";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Good {new Date().getHours() < 12 ? 'Morning' : 'Afternoon'}, {user.name}</h1>
        <p className="text-muted-foreground">Here is your wellbeing summary for today.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Wellbeing Score</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{Math.round(score * 100)} / 100</div>
            <p className="text-xs text-muted-foreground mt-1">Based on today's check-in</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Status</CardTitle>
            <HeartPulse className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
             <div className="text-2xl font-bold capitalize flex items-center gap-2">
               <span className={`inline-block w-3 h-3 rounded-full ${statusBg}`}></span>
               <span className={statusColor}>{status}</span>
             </div>
          </CardContent>
        </Card>

        <Card className="md:col-span-2 lg:col-span-1">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Dominant Emotion</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
               {emotionData.length > 0 ? emotionData.reduce((prev, current) => (prev.value > current.value) ? prev : current).name : "N/A"}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="col-span-4">
        <CardHeader>
          <CardTitle>Emotion Breakdown</CardTitle>
          <CardDescription>
            Your predicted emotional distribution for today.
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[300px]">
          {emotionData.length > 0 ? (
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={emotionData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                 <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                 <XAxis dataKey="name" axisLine={false} tickLine={false} />
                 <YAxis axisLine={false} tickLine={false} tickFormatter={(val) => `${val}%`} />
                 <Tooltip cursor={{fill: 'transparent'}} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                 <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                   {emotionData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={COLORS[entry.name] || "#000"} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          ) : (
             <div className="h-full flex items-center justify-center text-muted-foreground">
               No check-in data for today yet. Go to Daily Check-in to log your day.
             </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
