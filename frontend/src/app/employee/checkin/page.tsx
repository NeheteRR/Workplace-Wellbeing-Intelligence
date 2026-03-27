"use client";

import { useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { fetchWithAuth } from "@/lib/api-client";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Activity, AlertCircle, ArrowRight, CheckCircle2, HeartPulse, Send, Smile } from "lucide-react";

export default function CheckinPage() {
  const { user } = useAuth();
  const [reflection, setReflection] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reflection.trim() || reflection.length < 5) {
      setError("Please write at least a few words about your day.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await fetchWithAuth("/analyze-day", {
        method: "POST",
        body: JSON.stringify({
          org_id: user?.org_id,
          employee_id: user?.employee_id,
          text: reflection
        })
      });
      setResult(res);
      setReflection("");
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  if (!user) return null;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Daily Reflection</h1>
        <p className="text-muted-foreground">Take a moment to reflect on your workday.</p>
      </div>

      {!result ? (
        <Card>
          <form onSubmit={handleSubmit}>
            <CardHeader>
              <CardTitle>How are you feeling today?</CardTitle>
              <CardDescription>
                Write openly about your experiences, challenges, and successes. This helps us understand your wellbeing and provide tailored support.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {error && (
                <div className="flex items-center gap-2 p-3 text-sm text-red-600 bg-red-50 dark:bg-red-900/50 dark:text-red-200 rounded-md">
                  <AlertCircle className="w-4 h-4" />
                  <p>{error}</p>
                </div>
              )}
              <Textarea 
                placeholder="I felt productive this morning but got stressed after the 2PM meeting..."
                className="min-h-[150px] resize-y"
                value={reflection}
                onChange={(e) => setReflection(e.target.value)}
                disabled={loading}
              />
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button type="submit" disabled={loading} className="w-full sm:w-auto">
                {loading ? (
                  <span className="flex items-center">Analyzing... <Activity className="ml-2 w-4 h-4 animate-spin" /></span>
                ) : (
                  <span className="flex items-center">Analyze My Day <Send className="ml-2 w-4 h-4" /></span>
                )}
              </Button>
            </CardFooter>
          </form>
        </Card>
      ) : (
        <div className="space-y-6 animate-in slide-in-from-bottom-4 fade-in duration-500">
          <Card className="border-green-200 dark:border-green-900 shadow-sm">
             <CardContent className="pt-6 flex flex-col items-center text-center space-y-2">
                <div className="w-12 h-12 bg-green-100 dark:bg-green-900 text-green-600 rounded-full flex items-center justify-center mb-2">
                   <CheckCircle2 className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-semibold">Check-in Complete</h3>
                <p className="text-muted-foreground max-w-md">Your reflection has been safely recorded and your daily wellbeing score updated.</p>
             </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                 <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    <HeartPulse className="w-4 h-4 mr-2" /> Daily Wellbeing Score
                 </CardTitle>
              </CardHeader>
              <CardContent>
                 <div className="text-3xl font-bold">{Math.round(result.wellbeing_score * 100)} / 100</div>
                 <div className="text-sm capitalize text-muted-foreground mt-1">Status: {result.wellbeing_status}</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                 <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    <Smile className="w-4 h-4 mr-2" /> Dominant Emotion
                 </CardTitle>
              </CardHeader>
              <CardContent>
                 <div className="text-3xl font-bold capitalize">{result.dominant_emotion || "N/A"}</div>
              </CardContent>
            </Card>
          </div>

          {(result.assistant_message || result.suggestions) && (
            <Card className="bg-blue-50/50 dark:bg-blue-950/20 border-blue-100 dark:border-blue-900">
              <CardHeader>
                 <CardTitle className="text-lg">Insights for You</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                 {result.assistant_message && (
                   <p className="text-slate-700 dark:text-slate-300 italic">"{result.assistant_message}"</p>
                 )}
                 
                 {result.suggestions && result.suggestions.length > 0 && (
                   <div className="pt-4 border-t border-blue-100 dark:border-blue-800">
                     <h4 className="font-medium mb-3">Recommended Actions:</h4>
                     <ul className="space-y-2">
                       {result.suggestions.map((sbg: any, i: number) => (
                         <li key={i} className="flex gap-2 text-sm text-slate-700 dark:text-slate-300">
                           <ArrowRight className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                           <span><strong>{sbg.title}:</strong> {sbg.description}</span>
                         </li>
                       ))}
                     </ul>
                   </div>
                 )}
              </CardContent>
            </Card>
          )}

          <div className="flex justify-center pt-4">
            <Button variant="outline" onClick={() => setResult(null)}>Log Another Reflection</Button>
          </div>
        </div>
      )}
    </div>
  );
}
