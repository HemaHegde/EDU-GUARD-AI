import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip,
  PieChart, Pie, Cell, AreaChart, Area, CartesianGrid,
} from "recharts";
import { Users, AlertTriangle, Activity, Eye, Sparkles } from "lucide-react";
import { dashboardService } from "@/lib/api/services";
import { useApi } from "@/hooks/useApi";
import { PageShell, Skeleton, ErrorState } from "@/components/eg/PageShell";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Psychological Engagement Monitor · EduGuard-AI" },
      { name: "description", content: "Overview of learner engagement, disengagement risk, and psychological profiles — powered by the EduGuard-AI research backend." },
    ],
  }),
  component: DashboardPage,
});

type Overview = {
  total_students?: number;
  high_risk_students?: number;
  average_engagement?: number;
  average_attention?: number;
};

const PALETTE = ["oklch(0.78 0.13 330)", "oklch(0.78 0.1 290)", "oklch(0.82 0.09 235)", "oklch(0.84 0.1 165)", "oklch(0.86 0.1 55)"];

function DashboardPage() {
  const overview = useApi<Overview>(() => dashboardService.getOverview());
  const trend = useApi<Array<{ date: string; engagement: number; attention: number }>>(
    () => dashboardService.getEngagementTrend(),
  );
  const personas = useApi<Array<{ name: string; value: number }>>(
    () => dashboardService.getPersonaDistribution(),
  );

  return (
    <PageShell
      title="Psychological Engagement Monitor"
      description="Real-time overview of learner engagement metrics, disengagement risk indicators, and psychological profiles — streamed from the EduGuard-AI research backend."
    >
      <StatGrid loading={overview.loading} error={overview.error} data={overview.data} onRetry={overview.refetch} />

      <div className="grid gap-6 lg:grid-cols-3">
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-3xl p-6 lg:col-span-2"
        >
          <div className="mb-4 flex items-center justify-between">
            <div>
              <div className="text-sm text-muted-foreground">Engagement & attention</div>
              <div className="text-lg font-semibold">Cohort trend</div>
            </div>
            <span className="rounded-full bg-primary/15 px-3 py-1 text-xs font-medium text-primary">Live</span>
          </div>
          <div className="h-72">
            {trend.loading ? (
              <Skeleton className="h-full w-full" />
            ) : trend.error ? (
              <ErrorState message={trend.error} onRetry={trend.refetch} />
            ) : !trend.data || trend.data.length === 0 ? (
              <div className="grid h-full place-items-center text-sm text-muted-foreground">
                Connect your backend to see live engagement trends.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trend.data}>
                  <defs>
                    <linearGradient id="eg-eng" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="oklch(0.78 0.13 330)" stopOpacity={0.55} />
                      <stop offset="100%" stopColor="oklch(0.78 0.13 330)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="eg-att" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="oklch(0.78 0.1 290)" stopOpacity={0.5} />
                      <stop offset="100%" stopColor="oklch(0.78 0.1 290)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.9 0.02 320 / 0.4)" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="oklch(0.55 0.04 290)" />
                  <YAxis tick={{ fontSize: 11 }} stroke="oklch(0.55 0.04 290)" />
                  <Tooltip contentStyle={{ borderRadius: 16, border: "1px solid oklch(0.9 0.02 320)", background: "white" }} />
                  <Area type="monotone" dataKey="engagement" stroke="oklch(0.68 0.16 330)" strokeWidth={2.5} fill="url(#eg-eng)" />
                  <Area type="monotone" dataKey="attention" stroke="oklch(0.62 0.13 290)" strokeWidth={2.5} fill="url(#eg-att)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
          className="glass-card rounded-3xl p-6"
        >
          <div className="mb-4">
            <div className="text-sm text-muted-foreground">Persona distribution</div>
            <div className="text-lg font-semibold">Learner makeup</div>
          </div>
          <div className="h-72">
            {personas.loading ? (
              <Skeleton className="h-full w-full" />
            ) : personas.error ? (
              <ErrorState message={personas.error} onRetry={personas.refetch} />
            ) : !personas.data || personas.data.length === 0 ? (
              <div className="grid h-full place-items-center text-sm text-muted-foreground">
                No persona data yet.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={personas.data} dataKey="value" nameKey="name" innerRadius={55} outerRadius={95} paddingAngle={4}>
                    {personas.data.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 16, border: "1px solid oklch(0.9 0.02 320)", background: "white" }} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </motion.div>
      </div>
    </PageShell>
  );
}

function StatGrid({
  loading, error, data, onRetry,
}: { loading: boolean; error: string | null; data: Overview | null; onRetry: () => void }) {
  const stats = [
    { key: "total_students", label: "Total students", icon: Users, tint: "var(--pink)" },
    { key: "high_risk_students", label: "Disengagement risk flagged", icon: AlertTriangle, tint: "var(--peach)" },
    { key: "average_engagement", label: "Mean behavioral engagement", icon: Activity, tint: "var(--mint)", suffix: "%" },
    { key: "average_attention", label: "Mean attentional score", icon: Eye, tint: "var(--sky)", suffix: "%" },
  ] as const;

  if (error) return <ErrorState message={error} onRetry={onRetry} />;

  return (
    <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((s, i) => {
        const value = data?.[s.key as keyof Overview];
        const Icon = s.icon;
        return (
          <motion.div
            key={s.key}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="glass-card group relative overflow-hidden rounded-3xl p-5"
          >
            <div className="absolute -top-10 -right-10 h-32 w-32 rounded-full opacity-50 blur-2xl transition-opacity group-hover:opacity-80"
              style={{ background: `color-mix(in oklab, ${s.tint} 60%, transparent)` }} />
            <div className="relative flex items-start justify-between">
              <div className="text-xs uppercase tracking-wider text-muted-foreground">{s.label}</div>
              <span className="grid h-9 w-9 place-items-center rounded-xl"
                style={{ background: `color-mix(in oklab, ${s.tint} 35%, transparent)` }}>
                <Icon className="h-[18px] w-[18px]" />
              </span>
            </div>
            <div className="relative mt-5 h-9">
              {loading ? (
                <Skeleton className="h-full w-2/3" />
              ) : value === undefined || value === null ? (
                <div className="text-sm text-muted-foreground">No data</div>
              ) : (
                <div className="text-3xl font-semibold tracking-tight">
                  {typeof value === "number" ? value.toLocaleString() : String(value)}
                  {"suffix" in s && s.suffix ? <span className="ml-1 text-base text-muted-foreground">{s.suffix}</span> : null}
                </div>
              )}
            </div>
            <div className="relative mt-3 flex items-center gap-1.5 text-xs text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              Synced from FastAPI
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
