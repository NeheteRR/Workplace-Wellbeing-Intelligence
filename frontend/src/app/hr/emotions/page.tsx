"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";

export default function HREmotionsPage() {
  const { user } = useAuth();
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetchWithAuth(`/org/emotion-distribution`);
        setData(res);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const COLORS: Record<string, string> = {
    Joy: "#eab308",
    Sadness: "#3b82f6",
    Fear: "#a855f7",
    Anger: "#ef4444",
    Love: "#ec4899",
    Neutral: "#94a3b8"
  };

  if (!user) return null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Emotion Distribution</h1>
        <p className="text-muted-foreground">Aggregated emotional landscape across the organization.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Overall Emotion Breakdown</CardTitle>
          <CardDescription>
            Proportion of predicted emotions from all employee check-ins over the current logging period.
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[400px] flex items-center justify-center">
          {loading ? (
             <div className="h-64 w-64 rounded-full bg-slate-100 dark:bg-slate-800 animate-pulse" />
          ) : data.length > 0 ? (
             <ResponsiveContainer width="100%" height="100%">
               <PieChart>
                 <Pie
                   data={data}
                   cx="50%"
                   cy="50%"
                   innerRadius={80}
                   outerRadius={140}
                   paddingAngle={5}
                   dataKey="value"
                   label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}
                 >
                   {data.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={COLORS[entry.name] || "#000"} />
                   ))}
                 </Pie>
                 <Tooltip 
                    formatter={(value: any) => [`${(value as number * 100).toFixed(1)}%`, 'Share']}
                    contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                 />
                 <Legend verticalAlign="bottom" height={36}/>
               </PieChart>
             </ResponsiveContainer>
          ) : (
             <div className="text-muted-foreground">No emotion data available.</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
