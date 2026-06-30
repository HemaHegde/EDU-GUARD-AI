import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import {
  FlaskConical, Brain, BarChart3, TrendingUp, Target,
  Sparkles, BookOpen, Activity, Shield, AlertTriangle,
  ChevronRight, Beaker, GraduationCap, Layers,
} from "lucide-react";
import { PageShell, Skeleton, ErrorState } from "@/components/eg/PageShell";
import { api } from "@/lib/api/client";

export const Route = createFileRoute("/research")({
  head: () => ({
    meta: [
      { title: "Research Analytics · EduGuard-AI" },
      { name: "description", content: "Psychology research analytics dashboard — correlations, SHAP importance, model metrics, and persona profiles." },
    ],
  }),
  component: ResearchPage,
});

// ── types ────────────────────────────────────────────────────────────────────

type Overview = {
  title: string;
  subtitle: string;
  dataset: string;
  n_students: number;
  n_features: number;
  n_persona_clusters: number;
  psychological_theories: string[];
  ml_models: string[];
  model_accuracy: number;
  roc_auc: number;
};

type ModelMetrics = {
  metrics: { accuracy: number; precision: number; recall: number; f1_score: number; roc_auc: number };
  feature_importance: { feature: string; label: string; importance: number }[];
  clustering: { silhouette_score: number; davies_bouldin_index: number };
};

type Correlation = { feature: string; r: number; p: string; significance: string };
type CorrelationData = { note: string; correlations: Correlation[]; interpretation: string };

type ShapFeature = { feature: string; label: string; mean_shap: number; direction: string; psychological_construct: string };
type ShapData = { title: string; description: string; features: ShapFeature[]; interpretation: string };

type PersonaProfile = {
  persona: string; theory: string; description: string;
  intervention: string; psychological_constructs: string[];
};

type AnovaResult = {
  feature: string; f_statistic: number; p_value: string;
  significance: string; eta_squared: number; effect_size: string;
};
type AnovaData = { title: string; description: string; results: AnovaResult[]; interpretation: string };

// ── animation ────────────────────────────────────────────────────────────────

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: (i = 0) => ({
    opacity: 1, y: 0,
    transition: { duration: 0.55, delay: i * 0.07, ease: [0.22, 1, 0.36, 1] },
  }),
};
const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.07 } } };

// ── custom hook ──────────────────────────────────────────────────────────────

function useResearch<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function fetch() {
    try {
      setLoading(true);
      setError(null);
      const res = await api.get(`/research${path}`);
      setData(res.data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetch(); }, []);
  return { data, loading, error, refetch: fetch };
}

// ── palette ──────────────────────────────────────────────────────────────────

const METRIC_COLORS = {
  accuracy: "#7c3aed", precision: "#0ea5e9", recall: "#10b981",
  f1_score: "#f59e0b", roc_auc: "#f43f5e",
};

const SHAP_PROTECTIVE = "#10b981";
const SHAP_RISK = "#ef4444";

const PERSONA_COLORS: Record<string, string> = {
  "Burnout Pattern": "#ef4444",
  "Passive Watcher": "#64748b",
  "Anxiety-Spike Learner": "#f59e0b",
  "Consistent Learner": "#10b981",
  "Last-Minute Survivor": "#8b5cf6",
  "Silent Isolator": "#6366f1",
};

// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

// ── Study Overview ───────────────────────────────────────────────────────────

function StudyOverview({ data }: { data: Overview }) {
  const pills = [
    { label: `n = ${data.n_students.toLocaleString()}`, icon: GraduationCap, color: "#7c3aed" },
    { label: `${data.n_features} Features`, icon: Layers, color: "#0ea5e9" },
    { label: `${data.n_persona_clusters} Clusters`, icon: Brain, color: "#f43f5e" },
    { label: `AUC ${data.roc_auc}`, icon: Target, color: "#10b981" },
  ];

  return (
    <motion.div variants={fadeUp} custom={0} className="relative overflow-hidden rounded-3xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-7">
      <div className="pointer-events-none absolute -right-20 -top-20 h-72 w-72 rounded-full bg-violet-400/10 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-16 -left-16 h-56 w-56 rounded-full bg-sky-400/8 blur-3xl" />

      <div className="relative z-10">
        <div className="flex items-center gap-3 mb-2">
          <div className="rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 p-3 shadow-lg shadow-violet-200">
            <FlaskConical className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.15em] text-violet-600">Research Study</div>
            <div className="text-lg font-bold text-slate-900 leading-tight">{data.title}</div>
          </div>
        </div>
        <p className="text-sm text-slate-500 mt-1 mb-5">{data.subtitle} — {data.dataset}</p>

        <div className="flex flex-wrap gap-2">
          {pills.map(({ label, icon: Icon, color }) => (
            <div key={label} className="flex items-center gap-1.5 rounded-full px-3 py-1.5 border border-slate-200 bg-white/80 shadow-sm">
              <Icon className="h-3.5 w-3.5" style={{ color }} />
              <span className="text-xs font-semibold text-slate-700">{label}</span>
            </div>
          ))}
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          {data.psychological_theories.map((t) => (
            <span key={t} className="rounded-lg bg-violet-50 border border-violet-200 px-2.5 py-1 text-[11px] font-semibold text-violet-700">{t}</span>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

// ── Model Performance Cards ──────────────────────────────────────────────────

function ModelPerformanceCards({ data }: { data: ModelMetrics }) {
  const metrics = [
    { key: "accuracy",  label: "Accuracy",  value: data.metrics.accuracy },
    { key: "precision", label: "Precision",  value: data.metrics.precision },
    { key: "recall",    label: "Recall",     value: data.metrics.recall },
    { key: "f1_score",  label: "F1 Score",   value: data.metrics.f1_score },
    { key: "roc_auc",   label: "ROC-AUC",    value: data.metrics.roc_auc },
  ];

  return (
    <motion.div variants={fadeUp} custom={1}>
      <div className="mb-3">
        <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">XGBoost Classifier</div>
        <div className="text-lg font-bold text-slate-900">Model Performance Metrics</div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {metrics.map(({ key, label, value }, i) => {
          const color = METRIC_COLORS[key as keyof typeof METRIC_COLORS] || "#7c3aed";
          const pct = (value * 100).toFixed(1);
          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 + i * 0.06 }}
              className="group relative overflow-hidden rounded-2xl border border-slate-200 bg-white/70 backdrop-blur-xl shadow-sm p-4 text-center"
            >
              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300" style={{ background: `radial-gradient(circle at 50% 50%, ${color}08, transparent 70%)` }} />
              <div className="relative z-10 text-3xl font-black" style={{ color }}>{pct}%</div>
              <div className="relative z-10 text-xs font-semibold text-slate-500 mt-1">{label}</div>
              <motion.div
                className="mt-2 mx-auto h-1 rounded-full"
                style={{ background: color }}
                initial={{ width: 0 }}
                animate={{ width: `${value * 100}%` }}
                transition={{ duration: 1, delay: 0.4 + i * 0.06, ease: [0.22, 1, 0.36, 1] }}
              />
            </motion.div>
          );
        })}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <div className="rounded-2xl border border-slate-200 bg-white/70 backdrop-blur-xl shadow-sm p-4 text-center">
          <div className="text-2xl font-black text-indigo-600">{data.clustering.silhouette_score}</div>
          <div className="text-xs font-semibold text-slate-500 mt-1">Silhouette Score (K-Means)</div>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white/70 backdrop-blur-xl shadow-sm p-4 text-center">
          <div className="text-2xl font-black text-pink-600">{data.clustering.davies_bouldin_index}</div>
          <div className="text-xs font-semibold text-slate-500 mt-1">Davies-Bouldin Index</div>
        </div>
      </div>
    </motion.div>
  );
}

// ── Correlation Table ────────────────────────────────────────────────────────

function CorrelationSection({ data }: { data: CorrelationData }) {
  return (
    <motion.div variants={fadeUp} custom={2} className="rounded-3xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-6">
      <div className="flex items-center gap-2.5 mb-1">
        <div className="rounded-xl bg-sky-100 p-2">
          <BarChart3 className="h-4 w-4 text-sky-600" />
        </div>
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Table 2</div>
          <div className="text-base font-bold text-slate-900">Behavioral Feature Correlations</div>
        </div>
      </div>
      <p className="text-xs text-slate-500 mb-4">{data.note}</p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="text-left py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Feature</th>
              <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">r</th>
              <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">p-value</th>
              <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Sig.</th>
            </tr>
          </thead>
          <tbody>
            {data.correlations.map((c, i) => (
              <motion.tr
                key={c.feature}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.04 }}
                className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors"
              >
                <td className="py-2.5 px-3 font-medium text-slate-800">{c.feature}</td>
                <td className="py-2.5 px-3 text-center">
                  <span className={`font-bold ${c.r < 0 ? "text-emerald-600" : "text-red-500"}`}>{c.r.toFixed(3)}</span>
                </td>
                <td className="py-2.5 px-3 text-center text-slate-500">{c.p}</td>
                <td className="py-2.5 px-3 text-center font-bold text-amber-600">{c.significance}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 rounded-xl bg-sky-50 border border-sky-100 p-3">
        <p className="text-xs text-slate-600 leading-relaxed">{data.interpretation}</p>
      </div>
    </motion.div>
  );
}

// ── SHAP Feature Importance ──────────────────────────────────────────────────

function ShapSection({ data }: { data: ShapData }) {
  const chartData = data.features.map((f) => ({
    name: f.label,
    value: f.mean_shap,
    direction: f.direction,
    construct: f.psychological_construct,
  }));

  return (
    <motion.div variants={fadeUp} custom={3} className="rounded-3xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-6">
      <div className="flex items-center gap-2.5 mb-1">
        <div className="rounded-xl bg-emerald-100 p-2">
          <TrendingUp className="h-4 w-4 text-emerald-600" />
        </div>
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Figure 1</div>
          <div className="text-base font-bold text-slate-900">SHAP Feature Importance</div>
        </div>
      </div>
      <p className="text-xs text-slate-500 mb-4">{data.description}</p>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
            <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} axisLine={{ stroke: "#e2e8f0" }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: "#475569", fontWeight: 600 }} width={170} axisLine={false} />
            <Tooltip
              formatter={(value: number, _: string, props: any) => [
                `${value.toFixed(3)} — ${props.payload.construct}`,
                "Mean |SHAP|",
              ]}
              contentStyle={{
                borderRadius: 12, border: "1px solid #e2e8f0",
                background: "#fff", fontSize: 12, boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
              }}
            />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={20}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.direction === "protective" ? SHAP_PROTECTIVE : SHAP_RISK} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
        <div className="flex items-center gap-1.5">
          <div className="h-2.5 w-2.5 rounded-full" style={{ background: SHAP_PROTECTIVE }} />
          Protective Factor
        </div>
        <div className="flex items-center gap-1.5">
          <div className="h-2.5 w-2.5 rounded-full" style={{ background: SHAP_RISK }} />
          Risk Factor
        </div>
      </div>

      <div className="mt-3 rounded-xl bg-emerald-50 border border-emerald-100 p-3">
        <p className="text-xs text-slate-600 leading-relaxed">{data.interpretation}</p>
      </div>
    </motion.div>
  );
}

// ── Persona Profiles Table ───────────────────────────────────────────────────

function PersonaProfilesSection({ data }: { data: PersonaProfile[] }) {
  return (
    <motion.div variants={fadeUp} custom={4} className="rounded-3xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-6">
      <div className="flex items-center gap-2.5 mb-4">
        <div className="rounded-xl bg-pink-100 p-2">
          <Brain className="h-4 w-4 text-pink-600" />
        </div>
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Table 4</div>
          <div className="text-base font-bold text-slate-900">Learner Psychological Profiles</div>
        </div>
      </div>

      <div className="space-y-3">
        {data.map((p, i) => {
          const color = PERSONA_COLORS[p.persona] || "#7c3aed";
          return (
            <motion.div
              key={p.persona}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.06 }}
              className="rounded-2xl border border-slate-200 bg-white/80 p-4 hover:shadow-md transition-shadow"
              style={{ borderLeft: `3px solid ${color}` }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="rounded-lg px-2.5 py-1 text-xs font-bold text-white" style={{ background: color }}>
                    {p.persona}
                  </span>
                </div>
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                  {p.theory.split("—")[0].trim()}
                </span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed mb-2">{p.description}</p>
              <div className="flex flex-wrap gap-1.5 mb-2">
                {p.psychological_constructs.map((c) => (
                  <span key={c} className="rounded-md bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600">{c}</span>
                ))}
              </div>
              <div className="flex items-center gap-1.5 text-xs text-violet-600">
                <Sparkles className="h-3 w-3" />
                <span className="font-medium">{p.intervention}</span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

// ── ANOVA Table ──────────────────────────────────────────────────────────────

function AnovaSection({ data }: { data: AnovaData }) {
  return (
    <motion.div variants={fadeUp} custom={5} className="rounded-3xl border border-white/50 bg-white/70 backdrop-blur-xl shadow-xl p-6">
      <div className="flex items-center gap-2.5 mb-1">
        <div className="rounded-xl bg-amber-100 p-2">
          <Activity className="h-4 w-4 text-amber-600" />
        </div>
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Table 5</div>
          <div className="text-base font-bold text-slate-900">Cluster ANOVA Validation</div>
        </div>
      </div>
      <p className="text-xs text-slate-500 mb-4">{data.description}</p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="text-left py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Feature</th>
              <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">F</th>
              <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">p</th>
              <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">η²</th>
              <th className="text-center py-2 px-3 text-xs font-semibold text-slate-500 uppercase">Effect</th>
            </tr>
          </thead>
          <tbody>
            {data.results.map((r, i) => (
              <motion.tr
                key={r.feature}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.04 }}
                className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors"
              >
                <td className="py-2.5 px-3 font-medium text-slate-800">{r.feature}</td>
                <td className="py-2.5 px-3 text-center font-bold text-violet-600">{r.f_statistic.toFixed(2)}</td>
                <td className="py-2.5 px-3 text-center text-slate-500">{r.p_value}</td>
                <td className="py-2.5 px-3 text-center font-semibold text-slate-700">{r.eta_squared.toFixed(3)}</td>
                <td className="py-2.5 px-3 text-center">
                  <span className={`rounded-md px-2 py-0.5 text-[10px] font-bold ${
                    r.effect_size === "Large" ? "bg-red-50 text-red-600" :
                    r.effect_size === "Medium" ? "bg-amber-50 text-amber-600" :
                    "bg-slate-100 text-slate-600"
                  }`}>
                    {r.effect_size}
                  </span>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 rounded-xl bg-amber-50 border border-amber-100 p-3">
        <p className="text-xs text-slate-600 leading-relaxed">{data.interpretation}</p>
      </div>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

function ResearchPage() {
  const overview = useResearch<Overview>("/overview");
  const modelMetrics = useResearch<ModelMetrics>("/model-metrics");
  const correlations = useResearch<CorrelationData>("/correlations");
  const shap = useResearch<ShapData>("/shap-importance");
  const personas = useResearch<PersonaProfile[]>("/persona-profiles");
  const anova = useResearch<AnovaData>("/cluster-anova");

  const anyLoading = overview.loading || modelMetrics.loading || correlations.loading || shap.loading || personas.loading || anova.loading;
  const firstError = overview.error || modelMetrics.error || correlations.error || shap.error || personas.error || anova.error;

  return (
    <PageShell
      title="Research Analytics"
      description="Statistical analysis and model validation for the psychology research paper — all data computed from the OULAD dataset."
    >
      {anyLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-40" />
          <Skeleton className="h-32" />
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      ) : firstError ? (
        <ErrorState message={firstError} onRetry={() => {
          overview.refetch();
          modelMetrics.refetch();
          correlations.refetch();
          shap.refetch();
          personas.refetch();
          anova.refetch();
        }} />
      ) : (
        <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-5">
          {overview.data && <StudyOverview data={overview.data} />}
          {modelMetrics.data && <ModelPerformanceCards data={modelMetrics.data} />}
          {correlations.data && <CorrelationSection data={correlations.data} />}
          {shap.data && <ShapSection data={shap.data} />}
          {personas.data && <PersonaProfilesSection data={personas.data} />}
          {anova.data && <AnovaSection data={anova.data} />}
        </motion.div>
      )}
    </PageShell>
  );
}
