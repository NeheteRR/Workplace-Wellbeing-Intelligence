"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { format, parseISO } from "date-fns";

export default function WellbeingHistoryPage() {
  const { user } = useAuth();
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      if (!user) return;
      try {
        const res = await fetchWithAuth(`/employee/wellbeing-history?employee_id=${user.employee_id}`);
        // Format dates
        const formatted = res.map((d: any) => ({
           ...d,
           displayDate: format(parseISO(d.date), "MMM dd")
        }));
        setData(formatted);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [user]);

  if (!user) return null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Wellbeing History</h1>
        <p className="text-muted-foreground">Track your emotional wellness over time.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Score History</CardTitle>
          <CardDescription>
            Your calculated wellbeing score based on daily reflections.
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[400px]">
          {loading ? (
             <div className="h-full w-full bg-slate-100 dark:bg-slate-800 animate-pulse rounded-md border" />
          ) : data.length > 0 ? (
             <ResponsiveContainer width="100%" height="100%">
               <LineChart data={data} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                 <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                 <XAxis 
                    dataKey="displayDate" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fontSize: 12 }} 
                    dy={10}
                 />
                 <YAxis 
                    domain={[0, 1]} 
                    axisLine={false} 
                    tickLine={false} 
                    tickFormatter={(val) => `${val * 100}`} 
                 />
                 <Tooltip 
                    contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} 
                    formatter={(value: any) => [`${Math.round(value as number * 100)} / 100`, 'Score']}
                    labelFormatter={(label) => `Date: ${label}`}
                 />
                 <Line 
                    type="monotone" 
                    dataKey="score" 
                    stroke="#3b82f6" 
                    strokeWidth={3} 
                    activeDot={{ r: 8 }} 
                 />
               </LineChart>
             </ResponsiveContainer>
          ) : (
             <div className="h-full flex items-center justify-center text-muted-foreground">
               No history available yet.
             </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
