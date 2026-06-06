import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  TrendingDown,
  Brain,
  Clock,
  Activity,
  MousePointerClick,
  Star,
  CalendarDays,
  ShieldAlert,
  Eye,
  Zap,
  CheckCircle2,
  Timer,
  Flame,
  User,
  Lightbulb,
  TrendingUp,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  PageShell,
  Skeleton,
  ErrorState,
  EmptyState,
} from "@/components/eg/PageShell";
import { supabase } from "@/lib/supabase";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

// =========================
// ROUTE
// =========================

export const Route = createFileRoute("/risk")({
  head: () => ({
    meta: [{ title: "Academic Risk · EduGuard-AI" }],
  }),
  component: RiskPage,
});

// =========================
// RISK COLOR
// =========================

function riskTint(score = 0) {
  if (score >= 70) return { color: "oklch(0.7 0.18 25)", label: "High" };
  if (score >= 40) return { color: "oklch(0.82 0.14 70)", label: "Medium" };
  return { color: "oklch(0.78 0.12 165)", label: "Low" };
}

function gaugeColor(score = 0) {
  if (score >= 70) return "#e05252";
  if (score >= 40) return "#e0a552";
  return "#52c07a";
}

// =========================
// REASON META
// =========================

function reasonMeta(reason: string, index: number) {
  const lower = reason.toLowerCase();
  if (lower.includes("inactiv")) return { color: "#e05252", bar: "#e05252", weight: 88 };
  if (lower.includes("declining") || lower.includes("slope")) return { color: "#e0a552", bar: "#e0a552", weight: 76 };
  if (lower.includes("performance") || lower.includes("score")) return { color: "#f97316", bar: "#f97316", weight: 72 };
  if (lower.includes("interaction") || lower.includes("click")) return { color: "#60a5fa", bar: "#60a5fa", weight: 65 };
  if (lower.includes("consist")) return { color: "#a78bfa", bar: "#a78bfa", weight: 58 };
  if (lower.includes("fluctuat") || lower.includes("variab")) return { color: "#f472b6", bar: "#f472b6", weight: 54 };
  if (lower.includes("stable")) return { color: "#34d399", bar: "#34d399", weight: 20 };
  const palette = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];
  return { color: palette[index % palette.length], bar: palette[index % palette.length], weight: Math.max(30, 80 - index * 8) };
}

// =========================
// PERSONA META
// =========================

type PersonaMeta = {
  icon: React.ElementType;
  color: string;
  gradient: string;
  description: string;
  profile: string;
  intervention: string;
  confidence: number;
};

function personaMeta(persona: string): PersonaMeta {
  const map: Record<string, PersonaMeta> = {
    "Passive Watcher": {
      icon: Eye,
      color: "#60a5fa",
      gradient: "from-blue-400/20 to-blue-600/10",
      description: "Prefers observing content before actively engaging.",
      profile: "Tends to watch lectures and read materials without actively participating in discussions or assessments. May struggle to convert passive knowledge into active performance.",
      intervention: "Encourage interactive activities, quizzes, and group discussions to shift from passive to active learning.",
      confidence: 87,
    },
    "Anxiety-Spike Learner": {
      icon: Zap,
      color: "#f472b6",
      gradient: "from-pink-400/20 to-pink-600/10",
      description: "Shows strong performance but fluctuating confidence and engagement.",
      profile: "Demonstrates bursts of high activity followed by disengagement. Often triggered by upcoming deadlines or assessments.",
      intervention: "Guided pacing, stress management techniques, and consistent check-ins to stabilise engagement patterns.",
      confidence: 83,
    },
    "Consistent Learner": {
      icon: CheckCircle2,
      color: "#34d399",
      gradient: "from-emerald-400/20 to-emerald-600/10",
      description: "Maintains stable participation and learning outcomes.",
      profile: "Engages reliably across sessions with predictable and positive learning behaviour. Rarely shows signs of disengagement.",
      intervention: "Provide advanced learning opportunities, enrichment activities, and mentorship pathways.",
      confidence: 94,
    },
    "Last-Minute Survivor": {
      icon: Timer,
      color: "#f59e0b",
      gradient: "from-amber-400/20 to-amber-600/10",
      description: "Surges in activity close to deadlines but disengages otherwise.",
      profile: "Activity spikes around assessments and submission dates. Long gaps of inactivity between deadlines indicate poor time management.",
      intervention: "Deadline management coaching, structured study schedules, and early-warning check-ins.",
      confidence: 79,
    },
    "Burnout Pattern": {
      icon: Flame,
      color: "#e05252",
      gradient: "from-red-400/20 to-red-600/10",
      description: "Shows signs of chronic exhaustion and declining engagement over time.",
      profile: "Significant and sustained drop in engagement, activity, and performance. High inactivity periods suggest the learner may be overwhelmed.",
      intervention: "Mental wellness intervention, reduced workload periods, and one-on-one counselling support.",
      confidence: 92,
    },
    "Silent Isolator": {
      icon: User,
      color: "#a78bfa",
      gradient: "from-violet-400/20 to-violet-600/10",
      description: "Minimal interaction with peers and course content.",
      profile: "Very low engagement across all interaction metrics. May be experiencing social isolation, technical barriers, or motivational issues.",
      intervention: "Social engagement support, peer mentoring, and proactive outreach from instructors.",
      confidence: 85,
    },
    "Struggling Learner": {
      icon: AlertTriangle,
      color: "#fb923c",
      gradient: "from-orange-400/20 to-orange-600/10",
      description: "Consistently below average performance with limited improvement.",
      profile: "Shows effort through activity and attendance but struggles to convert engagement into academic success.",
      intervention: "Academic tutoring, foundational content review, and adaptive learning pathway recommendations.",
      confidence: 81,
    },
  };

  return map[persona] ?? {
    icon: Brain,
    color: "#6366f1",
    gradient: "from-indigo-400/20 to-indigo-600/10",
    description: "Unique learning profile detected.",
    profile: "The AI model has identified a distinct behavioural pattern for this learner that requires personalised attention.",
    intervention: "General academic support and personalised coaching recommended.",
    confidence: 70,
  };
}

// =========================
// PROGRESS BAR
// =========================

function AnimatedBar({ value, color, delay = 0 }: { value: number; color: string; delay?: number }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-muted">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 0.8, ease: "easeOut", delay }}
        className="h-full rounded-full"
        style={{ background: color }}
      />
    </div>
  );
}

// =========================
// COMPONENT
// =========================

function RiskPage() {
  const [student, setStudent] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function fetchStudents() {
    try {
      setLoading(true);
      setError("");
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error("User not logged in");
      const response = await fetch(`http://127.0.0.1:8000/risk/me/${user.id}`);
      if (!response.ok) throw new Error(`API Error ${response.status}`);
      const data = await response.json();
      if (data.status === "error") throw new Error(data.message);
      setStudent(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchStudents(); }, []);

  return (
    <PageShell
      title="Academic Risk"
      description="Explainable risk signals for every learner — what's slipping, why, and what to do next."
    >
      {loading ? (
        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={fetchStudents} />
      ) : !student ? (
        <EmptyState
          icon={<AlertTriangle className="h-8 w-8" />}
          title="No students yet"
          hint="Backend returned no risk data."
        />
      ) : (
        <div className="space-y-5">

          {/* ========================= */}
          {/* SECTION 1 — KPI CARDS    */}
          {/* ========================= */}

          <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
            {[
              { label: "Average Score", value: student.avg_score, suffix: "%", icon: Star, from: "#f59e42", to: "#f97316", delay: 0.05 },
              { label: "Active Days", value: student.active_days, suffix: " days", icon: CalendarDays, from: "#34d399", to: "#059669", delay: 0.1 },
              { label: "Total Clicks", value: student.total_clicks, suffix: "", icon: MousePointerClick, from: "#60a5fa", to: "#2563eb", delay: 0.15 },
              { label: "Inactivity Days", value: student.inactivity_days, suffix: " days", icon: Activity, from: "#f87171", to: "#dc2626", delay: 0.2 },
            ].map((kpi) => {
              const Icon = kpi.icon;
              return (
                <motion.div
                  key={kpi.label}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: kpi.delay }}
                  whileHover={{ scale: 1.03, y: -3 }}
                  className="glass-card relative overflow-hidden rounded-3xl p-5 cursor-default"
                >
                  <div
                    className="absolute -top-6 -right-6 h-20 w-20 rounded-full opacity-30 blur-2xl"
                    style={{ background: `linear-gradient(135deg, ${kpi.from}, ${kpi.to})` }}
                  />
                  <div
                    className="mb-3 flex h-10 w-10 items-center justify-center rounded-2xl"
                    style={{ background: `linear-gradient(135deg, ${kpi.from}22, ${kpi.to}33)` }}
                  >
                    <Icon className="h-5 w-5" style={{ color: kpi.from }} />
                  </div>
                  <div className="text-2xl font-bold" style={{ color: kpi.from }}>
                    {kpi.value}{kpi.suffix}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">{kpi.label}</div>
                </motion.div>
              );
            })}
          </div>

          {/* ========================= */}
          {/* SECTION 2 — RADAR CHART  */}
          {/* ========================= */}

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
            className="glass-card rounded-3xl p-6"
          >
            <div className="mb-1 text-lg font-semibold">Learning Behaviour Radar</div>
            <div className="mb-5 text-sm text-muted-foreground">
              Visual representation of academic engagement patterns.
            </div>
            <ResponsiveContainer width="100%" height={340}>
              <RadarChart
                data={[
                  { metric: "Score", value: student.avg_score },
                  { metric: "Activity", value: student.active_days },
                  { metric: "Clicks", value: Math.min(student.total_clicks / 10, 100) },
                  { metric: "Consistency", value: Math.max(100 - student.inactivity_days, 0) },
                  { metric: "Engagement", value: Math.min(student.total_clicks / 5, 100) },
                ]}
                cx="50%" cy="50%" outerRadius="75%"
              >
                <PolarGrid stroke="oklch(0.88 0 0)" strokeDasharray="3 3" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: "oklch(0.55 0 0)" }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10, fill: "oklch(0.65 0 0)" }} tickCount={4} />
                <Radar name="Student" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} strokeWidth={2} />
                <Tooltip
                  contentStyle={{ borderRadius: "12px", border: "1px solid oklch(0.9 0 0)", fontSize: "12px", background: "white" }}
                  formatter={(value: any) => [`${Number(value).toFixed(1)}`, "Score"]}
                />
              </RadarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* ========================= */}
          {/* SECTION 3 — SIDE BY SIDE */}
          {/* AI EXPLAINABILITY + PERSONA */}
          {/* ========================= */}

          <div className="grid gap-5 lg:grid-cols-2">

            {/* ── AI EXPLAINABILITY ── */}
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
              className="rounded-3xl border border-border bg-gradient-to-br from-white/70 to-white/30 p-5 backdrop-blur-sm glass-card"
            >
              <div className="mb-1 flex items-center gap-2">
                <div
                  className="flex h-8 w-8 items-center justify-center rounded-xl"
                  style={{ background: "#6366f122" }}
                >
                  <ShieldAlert className="h-4 w-4" style={{ color: "#6366f1" }} />
                </div>
                <span className="text-base font-semibold">AI Explainability</span>
              </div>
              <div className="mb-4 text-xs text-muted-foreground">
                Factors contributing to the current academic risk prediction.
              </div>

              {/* model confidence bar */}
              <div className="mb-4 rounded-xl bg-primary/5 px-4 py-3">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-muted-foreground font-medium">Model Confidence</span>
                  <span className="font-bold" style={{ color: gaugeColor(student.risk_score) }}>
                    {(student.prediction_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <AnimatedBar
                  value={student.prediction_probability * 100}
                  color={`linear-gradient(90deg, ${gaugeColor(student.risk_score)}, #6366f1)`}
                  delay={0.5}
                />
              </div>

              {/* risk score row */}
              <div className="mb-4 flex items-center justify-between rounded-xl bg-muted/40 px-4 py-3">
                <div>
                  <div className="text-xs text-muted-foreground font-medium">Risk Score</div>
                  <div className="text-xl font-black mt-0.5" style={{ color: gaugeColor(student.risk_score) }}>
                    {student.risk_score} / 100
                  </div>
                </div>
                <span
                  className="rounded-full px-3 py-1 text-xs font-semibold"
                  style={{
                    background: `${gaugeColor(student.risk_score)}22`,
                    color: gaugeColor(student.risk_score),
                  }}
                >
                  {student.risk_level}
                </span>
              </div>

              {/* reasons */}
              <div className="space-y-3">
                {student.reasons?.map((reason: string, index: number) => {
                  const meta = reasonMeta(reason, index);
                  return (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.45 + index * 0.07 }}
                      className="rounded-xl border p-3"
                      style={{ borderColor: `${meta.color}33`, background: `${meta.color}08` }}
                    >
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          <div
                            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg"
                            style={{ background: `${meta.color}22` }}
                          >
                            <Brain className="h-3.5 w-3.5" style={{ color: meta.color }} />
                          </div>
                          <span className="text-xs font-medium leading-snug">{reason}</span>
                        </div>
                        <span className="shrink-0 text-xs font-bold" style={{ color: meta.color }}>
                          {meta.weight}%
                        </span>
                      </div>
                      <AnimatedBar value={meta.weight} color={meta.bar} delay={0.5 + index * 0.07} />
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>

            {/* ── AI PERSONA (PREMIUM) ── */}
            {(() => {
              const s = student;
              const pm = personaMeta(s.persona);
              const PersonaIcon = pm.icon;

              return (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="glass-card relative overflow-hidden rounded-3xl border border-border p-5 backdrop-blur-sm"
                >
                  {/* gradient orb */}
                  <div
                    className="pointer-events-none absolute -top-10 -right-10 h-40 w-40 rounded-full blur-3xl opacity-30"
                    style={{ background: pm.color }}
                  />

                  <div className="relative z-10">
                    {/* top: large icon + name + confidence ring */}
                    <div className="flex items-start justify-between gap-4 mb-4">
                      <div className="flex items-center gap-4">
                        {/* large colored icon circle */}
                        <div
                          className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl shadow-lg"
                          style={{
                            background: `linear-gradient(135deg, ${pm.color}25, ${pm.color}10)`,
                            border: `2px solid ${pm.color}30`,
                          }}
                        >
                          <PersonaIcon className="h-8 w-8" style={{ color: pm.color }} />
                        </div>
                        <div>
                          <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-0.5">
                            AI Persona
                          </div>
                          <div className="text-lg font-black leading-tight">{s.persona}</div>
                          <div className="mt-1">
                            <span
                              className="rounded-full px-2.5 py-0.5 text-[10px] font-semibold"
                              style={{ background: `${pm.color}18`, color: pm.color }}
                            >
                              Detected
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* confidence ring */}
                      <div className="shrink-0 w-20 h-20">
                        <CircularProgressbar
                          value={pm.confidence}
                          maxValue={100}
                          text={`${pm.confidence}%`}
                          styles={buildStyles({
                            textSize: "22px",
                            pathColor: pm.color,
                            textColor: pm.color,
                            trailColor: "#f0f0f0",
                            pathTransitionDuration: 0.9,
                          })}
                        />
                        <div className="text-center text-[9px] text-muted-foreground mt-1">Confidence</div>
                      </div>
                    </div>

                    {/* description quote */}
                    <div
                      className="mb-4 rounded-xl px-3 py-2.5 text-xs italic leading-relaxed"
                      style={{
                        background: `${pm.color}0d`,
                        borderLeft: `3px solid ${pm.color}`,
                        color: pm.color,
                      }}
                    >
                      "{pm.description}"
                    </div>

                    {/* learner profile + intervention */}
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <div className="rounded-xl border border-border bg-white/50 p-3 text-xs leading-relaxed text-muted-foreground">
                        <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-foreground">
                          <User className="h-3 w-3" style={{ color: pm.color }} />
                          Learner Profile
                        </div>
                        {pm.profile}
                      </div>
                      <div className="rounded-xl border border-border bg-white/50 p-3 text-xs leading-relaxed text-muted-foreground">
                        <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-foreground">
                          <Lightbulb className="h-3 w-3" style={{ color: pm.color }} />
                          Recommended Action
                        </div>
                        {pm.intervention}
                      </div>
                    </div>

                    {/* mini stats row */}
                    <div className="mt-4 grid grid-cols-3 gap-2">
                      {[
                        { label: "Score", value: `${student.avg_score}%`, icon: TrendingUp },
                        { label: "Clicks", value: student.total_clicks, icon: MousePointerClick },
                        { label: "Inactive", value: `${student.inactivity_days}d`, icon: Clock },
                      ].map(({ label, value, icon: MiniIcon }) => (
                        <div
                          key={label}
                          className="rounded-xl bg-muted/40 p-2.5 text-center"
                        >
                          <MiniIcon className="h-3.5 w-3.5 mx-auto mb-1 text-muted-foreground" />
                          <div className="text-sm font-bold">{value}</div>
                          <div className="text-[10px] text-muted-foreground">{label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              );
            })()}
          </div>

          {/* ========================= */}
          {/* SECTION 4 — RISK DETAIL  */}
          {/* (inactivity/engagement    */}
          {/*  score/active days grid)  */}
          {/* ========================= */}

          {(() => {
            const s = student;
            const t = riskTint(s.risk_score);
            return (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="glass-card relative overflow-hidden rounded-3xl p-5"
              >
                <div
                  className="absolute -top-12 -right-10 h-32 w-32 rounded-full opacity-40 blur-2xl"
                  style={{ background: `color-mix(in oklab, ${t.color} 55%, transparent)` }}
                />

                <div className="relative flex items-center justify-between mb-4">
                  <div className="text-base font-semibold">{s.student_name} — Risk Detail</div>
                  <span
                    className="rounded-full px-3 py-1 text-xs font-medium"
                    style={{
                      background: `color-mix(in oklab, ${t.color} 22%, transparent)`,
                      color: t.color,
                    }}
                  >
                    {t.label} risk
                  </span>
                </div>

                <div className="relative mb-4">
                  <div className="flex items-baseline justify-between text-xs text-muted-foreground mb-2">
                    <span>Risk score</span>
                    <span className="text-foreground font-semibold">{s.risk_score}/100</span>
                  </div>
                  <AnimatedBar
                    value={s.risk_score}
                    color={`linear-gradient(90deg, ${t.color}, oklch(0.78 0.13 330))`}
                    delay={0.55}
                  />
                </div>

                <div className="relative grid grid-cols-2 gap-3 text-xs">
                  <div className="rounded-xl bg-muted/60 p-3">
                    <div className="flex items-center gap-1.5 text-muted-foreground">
                      <Clock className="h-3 w-3" /> Inactivity
                    </div>
                    <div className="mt-1 text-sm font-medium">{s.inactivity_days} days</div>
                  </div>
                  <div className="rounded-xl bg-muted/60 p-3">
                    <div className="flex items-center gap-1.5 text-muted-foreground">
                      <TrendingDown className="h-3 w-3" /> Engagement
                    </div>
                    <div className="mt-1 text-sm font-medium">{s.total_clicks}</div>
                  </div>
                  <div className="rounded-xl bg-muted/60 p-3">
                    <div className="text-muted-foreground">Average Score</div>
                    <div className="mt-1 text-sm font-medium">{s.avg_score}</div>
                  </div>
                  <div className="rounded-xl bg-muted/60 p-3">
                    <div className="text-muted-foreground">Active Days</div>
                    <div className="mt-1 text-sm font-medium">{s.active_days}</div>
                  </div>
                  <div className="rounded-xl bg-muted/60 p-3 col-span-2">
                    <div className="text-muted-foreground">Prediction Probability</div>
                    <div className="mt-1 text-sm font-medium">
                      {(s.prediction_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })()}

        </div>
      )}
    </PageShell>
  );
}