"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { format, parseISO } from "date-fns";

export default function EmotionHistoryPage() {
  const { user } = useAuth();
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      if (!user) return;
      try {
        const res = await fetchWithAuth(`/employee/emotion-history?employee_id=${user.employee_id}`);
        setData(res);
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
        <h1 className="text-3xl font-bold tracking-tight">Emotion History</h1>
        <p className="text-muted-foreground">View your detailed emotional logs over time.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>History Log</CardTitle>
          <CardDescription>
            A breakdown of your predicted emotions from past check-ins. Values are displayed as percentages.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
             <div className="h-48 w-full bg-slate-100 dark:bg-slate-800 animate-pulse rounded-md border" />
          ) : data.length > 0 ? (
             <div className="rounded-md border">
               <Table>
                 <TableHeader>
                   <TableRow>
                     <TableHead>Date</TableHead>
                     <TableHead className="text-right">Joy</TableHead>
                     <TableHead className="text-right">Sadness</TableHead>
                     <TableHead className="text-right">Fear</TableHead>
                     <TableHead className="text-right">Anger</TableHead>
                     <TableHead className="text-right">Love</TableHead>
                     <TableHead className="text-right">Neutral</TableHead>
                   </TableRow>
                 </TableHeader>
                 <TableBody>
                   {data.map((row, i) => (
                     <TableRow key={i}>
                       <TableCell className="font-medium">
                         {row.date ? format(parseISO(row.date), "MMM dd, yyyy") : "N/A"}
                       </TableCell>
                       <TableCell className="text-right">{(row.joy * 100).toFixed(1)}%</TableCell>
                       <TableCell className="text-right">{(row.sadness * 100).toFixed(1)}%</TableCell>
                       <TableCell className="text-right">{(row.fear * 100).toFixed(1)}%</TableCell>
                       <TableCell className="text-right">{(row.anger * 100).toFixed(1)}%</TableCell>
                       <TableCell className="text-right">{(row.love * 100).toFixed(1)}%</TableCell>
                       <TableCell className="text-right">{(row.neutral * 100).toFixed(1)}%</TableCell>
                     </TableRow>
                   ))}
                 </TableBody>
               </Table>
             </div>
          ) : (
             <div className="py-8 text-center text-muted-foreground">
               No history available yet.
             </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
