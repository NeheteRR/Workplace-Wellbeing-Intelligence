"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function HRDepartmentsPage() {
  const { user } = useAuth();
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetchWithAuth(`/org/department-insights`);
        setData(res);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (!user) return null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Department Insights</h1>
        <p className="text-muted-foreground">Compare wellbeing metrics across different teams.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Department Breakdown</CardTitle>
          <CardDescription>
            Average wellbeing score and risk distribution grouped by department.
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
                     <TableHead>Department</TableHead>
                     <TableHead className="text-right">Avg Wellbeing (0-100)</TableHead>
                     <TableHead className="text-center">Status</TableHead>
                     <TableHead className="text-right">Employees at Risk</TableHead>
                   </TableRow>
                 </TableHeader>
                 <TableBody>
                   {data.map((row, i) => {
                     const score = Math.round(row.avg_wellbeing * 100);
                     const status = score >= 55 ? "Healthy" : score >= 30 ? "Moderate" : "At Risk";
                     const statusColor = status === "Healthy" ? "bg-green-100 text-green-800 dark:bg-green-900/30" : status === "Moderate" ? "bg-amber-100 text-amber-800 dark:bg-amber-900/30" : "bg-red-100 text-red-800 dark:bg-red-900/30";
                     
                     return (
                       <TableRow key={i}>
                         <TableCell className="font-medium">{row.department}</TableCell>
                         <TableCell className="text-right">{score}</TableCell>
                         <TableCell className="text-center">
                            <Badge variant="outline" className={`${statusColor} border-none`}>{status}</Badge>
                         </TableCell>
                         <TableCell className="text-right">
                            {row.employees_at_risk > 0 ? (
                               <span className="text-red-600 font-medium">{row.employees_at_risk}</span>
                            ) : (
                               <span className="text-slate-500">0</span>
                            )}
                         </TableCell>
                       </TableRow>
                     )
                   })}
                 </TableBody>
               </Table>
             </div>
          ) : (
             <div className="py-8 text-center text-muted-foreground">
               No department data available.
             </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
