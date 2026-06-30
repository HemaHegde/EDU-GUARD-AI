import React, { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Loader2 } from "lucide-react";
import axios from "axios";

interface ShapImpact {
  feature: string;
  value: number;
  shap_value: number;
  impact_direction: "positive" | "negative";
  impact_magnitude: number;
}

interface ExplanationData {
  status: string;
  base_value: number;
  shap_explanations: ShapImpact[];
}

export function PsychologicalDashboard({ userId }: { userId: string }) {
  const [data, setData] = useState<ExplanationData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchExplanation = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/persona/explain/${userId}`);
        if (response.data.status === "success") {
          setData(response.data);
        }
      } catch (error) {
        console.error("Failed to fetch psychological explanation", error);
      } finally {
        setLoading(false);
      }
    };

    if (userId) {
      fetchExplanation();
    }
  }, [userId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!data || !data.shap_explanations) {
    return (
      <Card className="w-full">
        <CardContent className="pt-6">
          <p className="text-muted-foreground text-center">No psychological explanation data available.</p>
        </CardContent>
      </Card>
    );
  }

  // Format data for Recharts
  const chartData = data.shap_explanations.map((item) => ({
    name: item.feature.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()),
    impact: parseFloat(item.shap_value.toFixed(4)),
    direction: item.impact_direction,
  }));

  return (
    <Card className="w-full shadow-lg border-primary/20">
      <CardHeader className="bg-primary/5 pb-4">
        <CardTitle className="text-xl font-bold tracking-tight text-primary">
          Psychological Factor Analysis (SHAP Explainability)
        </CardTitle>
        <CardDescription>
          Scientific breakdown of the behavioral factors driving this student's cognitive and psychological state.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="h-[350px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} width={120} />
              <Tooltip 
                formatter={(value: number) => [value, "SHAP Impact"]}
                contentStyle={{ borderRadius: '8px', border: '1px solid rgba(0,0,0,0.1)' }}
              />
              <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.impact > 0 ? "#ef4444" : "#22c55e"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-6 p-4 rounded-md bg-muted/50 border border-border/50 text-sm">
          <p>
            <strong>How to read this chart for your research:</strong> Features with red bars (positive SHAP values) are pushing the student 
            towards a higher psychological risk state (e.g., Burnout, Anxiety). Green bars (negative SHAP values) act as protective factors.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
