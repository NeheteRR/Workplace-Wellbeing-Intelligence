"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Users, Activity, AlertTriangle, CheckSquare } from "lucide-react";

export default function HRDashboard() {
  const { user } = useAuth();
  const [data, setData] = useState<{
    avg_score: number;
    overall_status: string;
    employee_count: number;
    risk_employees: number;
    checkin_rate: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetchWithAuth(`/org/summary`);
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
        <h1 className="text-3xl font-bold tracking-tight">Organization Overview</h1>
        <p className="text-muted-foreground">High-level summary of your company's emotional wellbeing.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Employees</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <div className="h-8 bg-slate-200 animate-pulse rounded w-1/2"></div> : (
              <div className="text-3xl font-bold">{data?.employee_count || 0}</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">Active accounts</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Wellbeing Score</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? <div className="h-8 bg-slate-200 animate-pulse rounded w-1/2"></div> : (
              <div className="text-3xl font-bold">{Math.round((data?.avg_score || 0) * 100)} / 100</div>
            )}
            <p className="text-xs text-muted-foreground mt-1 capitalize">Status: {data?.overall_status || 'Unknown'}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Employees at Risk</CardTitle>
            <AlertTriangle className={`h-4 w-4 ${data?.risk_employees && data.risk_employees > 0 ? 'text-red-500' : 'text-muted-foreground'}`} />
          </CardHeader>
          <CardContent>
             {loading ? <div className="h-8 bg-slate-200 animate-pulse rounded w-1/2"></div> : (
              <div className={`text-3xl font-bold ${data?.risk_employees && data.risk_employees > 0 ? 'text-red-600' : ''}`}>
                 {data?.risk_employees || 0}
              </div>
            )}
            <p className="text-xs text-muted-foreground mt-1">Require attention</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Check-in Rate</CardTitle>
            <CheckSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
             {loading ? <div className="h-8 bg-slate-200 animate-pulse rounded w-1/2"></div> : (
              <div className="text-3xl font-bold">{data?.checkin_rate || 0}%</div>
            )}
            <p className="text-xs text-muted-foreground mt-1">Active participants</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
