import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Brain,
  Heart,
  BookOpen,
  Sparkles,
  Zap,
  Eye,
  AlertTriangle,
  TrendingUp,
  MousePointer2,
  Activity,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Shield,
  Target,
  Gauge,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import {
  PageShell,
  Skeleton,
  ErrorState,
  EmptyState,
} from "@/components/eg/PageShell";

import { supabase } from "@/lib/supabase";

export const Route = createFileRoute("/persona")({
  head: () => ({
    meta: [{ title: "Persona Intelligence · EduGuard-AI" }],
  }),
  component: PersonaPage,
});

type PersonaData = {
  status: string;
  student_name: string;
  persona: string;
  attention_score: number;
  confusion_score: number;
  cognitive_overload_score: number;
  engagement_score: number;
  avg_score: number;
  active_days: number;
  total_clicks: number;
  cluster: number;
  intervention_style: string;
};

// ── helpers ──────────────────────────────────────────────────────────────────

function getPersonaIcon(persona: string) {
  const p = persona.toLowerCase();
  if (p.includes("anxiety")) return <Heart className="h-8 w-8 text-pink-500" />;
  if (p.includes("consistent")) return <Brain className="h-8 w-8 text-violet-500" />;
  return <BookOpen className="h-8 w-8 text-sky-500" />;
}

function getPersonaSummary(persona: string) {
  const p = persona.toLowerCase();
  if (p.includes("anxiety"))
    return "Shows strong academic potential but experiences fluctuating confidence and engagement levels. Benefits from structured reassurance and clear milestones.";
  if (p.includes("passive") || p.includes("watcher"))
    return "Prefers observing learning content before actively participating. Thrives with guided pacing and low-pressure interaction opportunities.";
  if (p.includes("consistent"))
    return "Maintains stable engagement and learning outcomes across sessions. Demonstrates disciplined habits and predictable performance curves.";
  return "Exhibits a unique learning signature that blends multiple cognitive patterns. Adaptive intervention strategies are recommended.";
}

function getPersonaColor(persona: string) {
  const p = persona.toLowerCase();
  if (p.includes("anxiety")) return { from: "#f43f5e", to: "#fb923c" };
  if (p.includes("consistent")) return { from: "#7c3aed", to: "#6366f1" };
  return { from: "#0ea5e9", to: "#06b6d4" };
}

function getStrengths(persona: string) {
  const p = persona.toLowerCase();
  if (p.includes("anxiety"))
    return ["Strong capability under pressure", "Fast learning bursts", "High assessment focus", "Driven by performance goals"];
  if (p.includes("passive") || p.includes("watcher"))
    return ["Deep content absorption", "Reflective thinking style", "Low error rate after observation", "Thorough comprehension"];
  if (p.includes("consistent"))
    return ["High self-discipline", "Stable engagement patterns", "Reliable performance", "Long-term retention"];
  return ["Adaptive learning style", "Cross-domain curiosity", "Resilient under changes", "Diverse engagement modes"];
}

function getBehavioralProfile(persona: string) {
  const p = persona.toLowerCase();
  if (p.includes("anxiety"))
    return "This learner exhibits high-intensity engagement spikes correlated with assessment events. Emotional regulation and confidence scaffolding are key levers. Cognitive load peaks before high-stakes moments, suggesting the need for pre-task preparation strategies.";
  if (p.includes("passive") || p.includes("watcher"))
    return "This learner demonstrates a watch-first, act-later pattern. Content consumption is thorough but interaction initiation is low. Nudges that reduce perceived risk of participation can significantly improve active engagement.";
  if (p.includes("consistent"))
    return "This learner maintains a steady rhythm of participation and performance. Engagement is predictable and sustainable. The primary growth opportunity lies in stretching into higher-order tasks and collaborative challenges.";
  return "This learner displays a multifaceted behavioral signature. Patterns vary by context, suggesting sensitivity to environmental and task-type factors. Personalized pathways outperform generic interventions for this profile.";
}

function getStatusInfo(value: number, type: "high" | "low") {
  if (type === "high") {
    if (value <= 30) return { label: "Healthy", color: "#22c55e", bg: "bg-emerald-50", text: "text-emerald-600", Icon: CheckCircle2 };
    if (value <= 60) return { label: "Monitor", color: "#f59e0b", bg: "bg-amber-50", text: "text-amber-600", Icon: AlertCircle };
    return { label: "Intervention Needed", color: "#ef4444", bg: "bg-red-50", text: "text-red-600", Icon: XCircle };
  } else {
    if (value >= 70) return { label: "Healthy", color: "#22c55e", bg: "bg-emerald-50", text: "text-emerald-600", Icon: CheckCircle2 };
    if (value >= 40) return { label: "Monitor", color: "#f59e0b", bg: "bg-amber-50", text: "text-amber-600", Icon: AlertCircle };
    return { label: "Intervention Needed", color: "#ef4444", bg: "bg-red-50", text: "text-red-600", Icon: XCircle };
  }
}

function getRiskLevel(persona: PersonaData) {
  const avg = (persona.attention_score + persona.engagement_score) / 2;
  const bad = (persona.confusion_score + persona.cognitive_overload_score) / 2;
  if (avg >= 65 && bad <= 35) return { label: "Low", color: "text-emerald-600", bg: "bg-emerald-50", dot: "bg-emerald-500" };
  if (avg >= 45 && bad <= 55) return { label: "Medium", color: "text-amber-600", bg: "bg-amber-50", dot: "bg-amber-500" };
  return { label: "High", color: "text-red-600", bg: "bg-red-50", dot: "bg-red-500" };
}

// ── animation variants ────────────────────────────────────────────────────────

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] },
  }),
};

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

// ── Progress Bar ──────────────────────────────────────────────────────────────

function ProgressBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200 overflow-hidden">
      <motion.div
        className="h-full rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(value, 100)}%` }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
      />
    </div>
  );
}

// ── Top Summary Row ───────────────────────────────────────────────────────────

function SummaryRow({ persona }: { persona: PersonaData }) {
  const risk = getRiskLevel(persona);
  const items = [
    { label: "Attention", value: persona.attention_score, unit: "", color: "#0ea5e9", Icon: Eye },
    { label: "Avg Score", value: persona.avg_score, unit: "", color: "#7c3aed", Icon: TrendingUp },
    { label: "Engagement", value: persona.engagement_score, unit: "", color: "#10b981", Icon: Zap },
    { label: "Risk Level", value: risk.label, unit: "", color: "#f59e0b", Icon: Gauge, isText: true, risk },
  ];

  return (
    <motion.div variants={fadeUp} custom={0} className="grid grid-cols-2 gap-3">
      {items.map(({ label, value, color, Icon, isText, risk: r }, i) => (
        <motion.div
          key={label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.07, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
          className="rounded-2xl border border-slate-200 bg-white/70 backdrop-blur-xl shadow-sm p-4"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{label}</div>
            <div className="rounded-lg p-1.5" style={{ background: `${color}18` }}>
              <Icon className="h-3.5 w-3.5" style={{ color }} />
            </div>
          </div>
          {isText && r ? (
            <div className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${r.dot}`} />
              <span className={`text-lg font-black ${r.color}`}>{value}</span>
            </div>
          ) : (
            <>
              <div className="text-2xl font-black text-slate-900">{value}</div>
              <ProgressBar value={Number(value)} color={color} />
            </>
          )}
        </motion.div>
      ))}
    </motion.div>
  );
}

// ── Hero Card ─────────────────────────────────────────────────────────────────

function HeroCard({ persona }: { persona: PersonaData }) {
  const colors = getPersonaColor(persona.persona);
  const summary = getPersonaSummary(persona.persona);

  return (
    <motion.div
      variants={fadeUp}
      custom={1}
      className="relative overflow-hidden rounded-3xl p-6 bg-white/70 backdrop-blur-xl border border-white/50 shadow-xl"
      style={{ borderTop: `3px solid ${colors.from}` }}
    >
      <div
        className="pointer-events-none absolute -right-16 -top-16 h-64 w-64 rounded-full blur-3xl"
        style={{ background: `radial-gradient(circle, ${colors.from}18, transparent 70%)` }}
      />
      <div
        className="pointer-events-none absolute -bottom-12 -left-12 h-48 w-48 rounded-full blur-3xl"
        style={{ background: `radial-gradient(circle, ${colors.to}12, transparent 70%)` }}
      />

      <div className="relative z-10">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">
              Learning Intelligence
            </div>
            <div className="mt-1 text-3xl font-black tracking-tight text-slate-900">
              {persona.student_name}
            </div>
          </div>
          <div
            className="rounded-2xl p-3"
            style={{ background: `${colors.from}15`, border: `1px solid ${colors.from}30` }}
          >
            {getPersonaIcon(persona.persona)}
          </div>
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <div
            className="rounded-2xl px-4 py-2 shadow-sm"
            style={{ background: `linear-gradient(135deg, ${colors.from}, ${colors.to})` }}
          >
            <span className="text-sm font-bold text-white">{persona.persona}</span>
          </div>
          <div className="flex items-center gap-1.5 rounded-2xl bg-slate-100 px-3 py-2">
            <Sparkles className="h-3.5 w-3.5 text-yellow-500" />
            <span className="text-xs font-semibold text-slate-700">92% Confidence</span>
          </div>
          <div className="flex items-center gap-1.5 rounded-2xl bg-slate-100 px-3 py-2">
            <Activity className="h-3.5 w-3.5 text-emerald-500" />
            <span className="text-xs font-semibold text-slate-700">{persona.active_days} Active Days</span>
          </div>
        </div>

        <p className="mt-5 leading-relaxed text-slate-600 text-sm">{summary}</p>
      </div>
    </motion.div>
  );
}

// ── Radar Section ─────────────────────────────────────────────────────────────

function RadarSection({ persona }: { persona: PersonaData }) {
  const data = [
    { subject: "Attention", value: persona.attention_score, fullMark: 100 },
    { subject: "Engagement", value: persona.engagement_score, fullMark: 100 },
    { subject: "Confusion", value: persona.confusion_score, fullMark: 100 },
    { subject: "Cognitive Load", value: persona.cognitive_overload_score, fullMark: 100 },
    { subject: "Acad. Score", value: persona.avg_score, fullMark: 100 },
  ];

  return (
    <motion.div
      variants={fadeUp}
      custom={2}
      className="rounded-3xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-6"
    >
      <div className="mb-1 text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">
        AI Analysis
      </div>
      <div className="text-lg font-bold text-slate-900">Cognitive Learning Profile</div>
      <div className="mt-0.5 text-xs text-slate-500">
        Multidimensional analysis of learning behaviour
      </div>

      <div className="mt-4 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
            <PolarGrid stroke="#CBD5E1" />
            <PolarAngleAxis
              dataKey="subject"
              tick={{ fill: "#475569", fontSize: 11, fontWeight: 600 }}
            />
            <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
            <Tooltip
              contentStyle={{
                background: "#fff",
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                color: "#1e293b",
                fontSize: 12,
                boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
              }}
            />
            <Radar
              name="Score"
              dataKey="value"
              stroke="#7c3aed"
              fill="#7c3aed"
              fillOpacity={0.15}
              strokeWidth={2}
              dot={{ fill: "#7c3aed", r: 3, strokeWidth: 0 }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}

// ── Learning DNA (KPI Cards with progress bars) ───────────────────────────────

const kpiCards = [
  { key: "attention_score", label: "Attention", Icon: Eye, grad: "from-sky-500 to-blue-600", barColor: "#0ea5e9" },
  { key: "engagement_score", label: "Engagement", Icon: Zap, grad: "from-violet-500 to-purple-600", barColor: "#7c3aed" },
  { key: "confusion_score", label: "Confusion", Icon: AlertTriangle, grad: "from-orange-500 to-amber-500", barColor: "#f59e0b" },
  { key: "cognitive_overload_score", label: "Cognitive Overload", Icon: Brain, grad: "from-rose-500 to-pink-600", barColor: "#f43f5e" },
  { key: "avg_score", label: "Average Score", Icon: TrendingUp, grad: "from-emerald-500 to-teal-500", barColor: "#10b981" },
  { key: "total_clicks", label: "Total Clicks", Icon: MousePointer2, grad: "from-cyan-500 to-sky-500", barColor: "#06b6d4" },
];

function LearningDNA({ persona }: { persona: PersonaData }) {
  return (
    <motion.div variants={stagger} initial="hidden" animate="show">
      <motion.div variants={fadeUp} custom={0} className="mb-3">
        <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Learning DNA</div>
        <div className="text-lg font-bold text-slate-900">Metric Breakdown</div>
      </motion.div>
      <div className="grid grid-cols-2 gap-3">
        {kpiCards.map(({ key, label, Icon, grad, barColor }, i) => {
          const value = (persona as any)[key];
          return (
            <motion.div
              key={key}
              variants={fadeUp}
              custom={i + 1}
              whileHover={{ y: -3, scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
              className="group relative overflow-hidden rounded-2xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-4"
            >
              {/* subtle hover tint */}
              <div
                className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${grad} opacity-0 group-hover:opacity-[0.07] transition-opacity duration-300`}
              />
              <div className={`relative z-10 inline-flex rounded-xl bg-gradient-to-br ${grad} p-2 shadow-md`}>
                <Icon className="h-4 w-4 text-white" />
              </div>
              <div className="relative z-10 mt-3 text-2xl font-black text-slate-900">{value}</div>
              <div className="relative z-10 mt-0.5 text-xs text-slate-500 font-medium">{label}</div>
              <div className="relative z-10">
                <ProgressBar value={Number(value)} color={barColor} />
                <div className="mt-1 text-[10px] text-slate-400 text-right">{value} / 100</div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

// ── Persona Insights ──────────────────────────────────────────────────────────

function PersonaInsights({ persona }: { persona: PersonaData }) {
  const strengths = getStrengths(persona.persona);
  const profile = getBehavioralProfile(persona.persona);

  return (
    <motion.div variants={fadeUp} custom={3}>
      <div className="mb-3">
        <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">AI Insights</div>
        <div className="text-lg font-bold text-slate-900">Persona Insights</div>
      </div>
      <div className="grid grid-cols-1 gap-3">
        <div className="rounded-2xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="rounded-xl bg-indigo-100 p-2">
              <Brain className="h-4 w-4 text-indigo-600" />
            </div>
            <span className="text-sm font-bold text-slate-900">Behavioral Profile</span>
          </div>
          <p className="text-sm leading-relaxed text-slate-600">{profile}</p>
        </div>

        <div className="rounded-2xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="rounded-xl bg-emerald-100 p-2">
              <Shield className="h-4 w-4 text-emerald-600" />
            </div>
            <span className="text-sm font-bold text-slate-900">Strengths</span>
          </div>
          <div className="space-y-2">
            {strengths.map((s, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + i * 0.07 }}
                className="flex items-center gap-2.5"
              >
                <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 flex-shrink-0" />
                <span className="text-sm text-slate-600">{s}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ── Intervention Card ─────────────────────────────────────────────────────────

function InterventionCard({ persona }: { persona: PersonaData }) {
  return (
    <motion.div
      variants={fadeUp}
      custom={4}
      className="relative overflow-hidden rounded-3xl border border-violet-200 bg-gradient-to-br from-violet-500/15 to-indigo-500/15 backdrop-blur-xl shadow-xl p-6"
    >
      <div className="pointer-events-none absolute -right-8 -top-8 h-40 w-40 rounded-full bg-violet-400/10 blur-2xl" />
      <div className="pointer-events-none absolute -bottom-8 -left-8 h-32 w-32 rounded-full bg-indigo-400/10 blur-2xl" />

      <div className="relative z-10">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-500 p-3 shadow-lg shadow-violet-200">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.15em] text-violet-600">AI Coach</div>
              <div className="text-base font-bold text-slate-900">Recommended Intervention</div>
            </div>
          </div>
          <div className="flex-shrink-0 rounded-full bg-violet-100 px-3 py-1 text-xs font-bold text-violet-700 border border-violet-200">
            HIGH PRIORITY
          </div>
        </div>

        <div className="mt-5 rounded-2xl bg-white/60 border border-violet-100 p-4 backdrop-blur-sm">
          <div className="flex items-start gap-2.5">
            <Target className="h-4 w-4 text-violet-500 mt-0.5 flex-shrink-0" />
            <p className="text-sm leading-relaxed text-slate-700">{persona.intervention_style}</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ── Learning Status ───────────────────────────────────────────────────────────

function LearningStatus({ persona }: { persona: PersonaData }) {
  const attentionStatus = getStatusInfo(persona.attention_score, "low");
  const engagementStatus = getStatusInfo(persona.engagement_score, "low");
  const overloadStatus = getStatusInfo(persona.cognitive_overload_score, "high");

  const statuses = [
    { label: "Attention Status", value: persona.attention_score, ...attentionStatus },
    { label: "Engagement Status", value: persona.engagement_score, ...engagementStatus },
    { label: "Cognitive Load Status", value: persona.cognitive_overload_score, ...overloadStatus },
  ];

  return (
    <motion.div variants={fadeUp} custom={5}>
      <div className="mb-3">
        <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Live Monitoring</div>
        <div className="text-lg font-bold text-slate-900">Learning Status</div>
      </div>
      <div className="grid grid-cols-1 gap-3">
        {statuses.map(({ label, value, label: statusLabel, bg, text, Icon, color }, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, x: -15 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 + i * 0.1 }}
            className={`flex items-center justify-between rounded-2xl border border-slate-200 ${bg} p-4`}
          >
            <div className="flex-1 mr-4">
              <div className="text-xs text-slate-500 font-medium">{label}</div>
              <div className={`mt-0.5 text-sm font-bold ${text}`}>{statusLabel}</div>
              <ProgressBar value={value} color={color} />
            </div>
            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="text-lg font-black text-slate-800">{value}</div>
                <div className="text-[10px] text-slate-400">/ 100</div>
              </div>
              <Icon className={`h-5 w-5 ${text}`} />
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

function PersonaPage() {
  const [persona, setPersona] = useState<PersonaData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function fetchPersona() {
    try {
      setLoading(true);
      setError("");

      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) throw new Error("User not logged in");

      const response = await fetch(`http://127.0.0.1:8000/persona/me/${user.id}`);

      if (!response.ok) throw new Error(`API Error ${response.status}`);

      const data = await response.json();

      if (data.status === "error") throw new Error(data.message);

      setPersona(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchPersona();
  }, []);

  return (
    <PageShell
      title="Persona Intelligence"
      description="AI-generated learning persona based on academic and cognitive behavior."
    >
      {loading ? (
        <Skeleton className="h-96" />
      ) : error ? (
        <ErrorState message={error} onRetry={fetchPersona} />
      ) : !persona ? (
        <EmptyState
          icon={<Brain className="h-8 w-8" />}
          title="No persona found"
          hint="Backend returned no persona data."
        />
      ) : (
        <motion.div
          variants={stagger}
          initial="hidden"
          animate="show"
          className="space-y-4"
        >
          {/* S0 — Top Summary Row */}
          <SummaryRow persona={persona} />

          {/* S1 — Hero */}
          <HeroCard persona={persona} />

          {/* S2 — Radar */}
          <RadarSection persona={persona} />

          {/* S3 — KPI DNA */}
          <LearningDNA persona={persona} />

          {/* S4 — Insights */}
          <PersonaInsights persona={persona} />

          {/* S5 — Intervention */}
          <InterventionCard persona={persona} />

          {/* S6 — Status */}
          <LearningStatus persona={persona} />
        </motion.div>
      )}
    </PageShell>
  );
}
