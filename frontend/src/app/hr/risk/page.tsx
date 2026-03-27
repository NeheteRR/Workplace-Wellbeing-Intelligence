"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle } from "lucide-react";

export default function HRRiskPage() {
  const { user } = useAuth();
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetchWithAuth(`/org/risk-employees`);
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
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
           <AlertTriangle className="text-red-500 h-8 w-8" /> 
           Risk Signals
        </h1>
        <p className="text-muted-foreground">Identify and support employees showing signs of low wellbeing or burnout.</p>
      </div>

      <Card className="border-red-200 dark:border-red-900/50">
        <CardHeader className="bg-red-50 dark:bg-red-950/20 rounded-t-lg">
          <CardTitle className="text-red-800 dark:text-red-400">Employees at Risk</CardTitle>
          <CardDescription>
            This list is auto-generated based on consistent patterns of low wellbeing scores or negative emotional markers. Raw text from check-ins is kept private.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          {loading ? (
             <div className="h-48 w-full bg-slate-100 dark:bg-slate-800 animate-pulse rounded-md border" />
          ) : data.length > 0 ? (
             <div className="rounded-md border">
               <Table>
                 <TableHeader>
                   <TableRow>
                     <TableHead>Employee</TableHead>
                     <TableHead>Department</TableHead>
                     <TableHead>Trend</TableHead>
                     <TableHead className="text-right">Risk Level</TableHead>
                     <TableHead className="text-center">Action</TableHead>
                   </TableRow>
                 </TableHeader>
                 <TableBody>
                   {data.map((row, i) => (
                     <TableRow key={i}>
                       <TableCell className="font-medium">{row.employee}</TableCell>
                       <TableCell>{row.department}</TableCell>
                       <TableCell>
                          <span className="flex items-center text-red-600 text-sm">
                             <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                             </svg>
                             {row.trend}
                          </span>
                       </TableCell>
                       <TableCell className="text-right">
                          <Badge variant="destructive">{row.risk_level}</Badge>
                       </TableCell>
                       <TableCell className="text-center">
                          <button className="text-sm font-medium text-blue-600 hover:underline">Schedule Check-in</button>
                       </TableCell>
                     </TableRow>
                   ))}
                 </TableBody>
               </Table>
             </div>
          ) : (
             <div className="py-12 flex flex-col items-center justify-center text-center">
               <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 text-green-600 flex items-center justify-center rounded-full mb-4">
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                     <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
               </div>
               <h3 className="text-xl font-semibold mb-2">Excellent News</h3>
               <p className="text-muted-foreground max-w-sm">
                 There are currently no employees flagged as high risk by the wellbeing monitoring system.
               </p>
             </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
