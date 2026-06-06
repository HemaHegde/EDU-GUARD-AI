import { Link, useRouterState } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useState } from "react";
import {
  LayoutDashboard,
  AlertTriangle,
  Brain,
  Users,
  Sparkles,
  BookOpen,
  Video,
  ChevronLeft,
} from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, color: "var(--pink)" },
  { to: "/risk", label: "Academic Risk", icon: AlertTriangle, color: "var(--peach)" },
  { to: "/cognitive", label: "Cognitive Intelligence", icon: Brain, color: "var(--lavender)" },
  { to: "/persona", label: "Persona Intelligence", icon: Users, color: "var(--sky)" },
  { to: "/mentor", label: "AI Mentor", icon: Sparkles, color: "var(--primary)" },
  { to: "/learn", label: "Quiz + Flashcards", icon: BookOpen, color: "var(--mint)" },
  { to: "/video", label: "Video Intelligence", icon: Video, color: "var(--secondary)" },
] as const;

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <motion.aside
      animate={{ width: collapsed ? 84 : 264 }}
      transition={{ type: "spring", stiffness: 220, damping: 28 }}
      className="sticky top-0 z-30 hidden h-screen shrink-0 flex-col gap-2 border-r border-border/60 bg-sidebar/70 p-4 backdrop-blur-xl md:flex"
    >
      <div className="flex items-center justify-between px-1 py-2">
        <Link to="/" className="flex items-center gap-2.5">
          <div className="grid h-10 w-10 place-items-center rounded-2xl gradient-aurora shadow-[var(--shadow-glow)]">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              className="leading-tight"
            >
              <div className="text-[15px] font-semibold tracking-tight text-gradient">
                EduGuard-AI
              </div>
              <div className="text-[11px] text-muted-foreground">Learning companion</div>
            </motion.div>
          )}
        </Link>
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="grid h-8 w-8 place-items-center rounded-full border border-border/60 bg-card/60 text-muted-foreground transition hover:bg-accent/40 hover:text-foreground"
          aria-label="Toggle sidebar"
        >
          <ChevronLeft className={cn("h-4 w-4 transition-transform", collapsed && "rotate-180")} />
        </button>
      </div>

      <nav className="mt-3 flex flex-col gap-1.5">
        {items.map((it) => {
          const active = it.to === "/" ? pathname === "/" : pathname.startsWith(it.to);
          const Icon = it.icon;
          return (
            <Link
              key={it.to}
              to={it.to}
              className={cn(
                "group relative flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm transition-all",
                active
                  ? "bg-white/70 text-foreground shadow-[0_8px_24px_-12px_oklch(0.7_0.15_320/0.45)] dark:bg-white/10"
                  : "text-muted-foreground hover:bg-accent/40 hover:text-foreground",
              )}
            >
              {active && (
                <motion.span
                  layoutId="sidebar-active"
                  className="absolute inset-0 -z-10 rounded-2xl gradient-soft"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <span
                className="grid h-9 w-9 shrink-0 place-items-center rounded-xl transition-all group-hover:scale-105"
                style={{
                  background: `color-mix(in oklab, ${it.color} 28%, transparent)`,
                  boxShadow: active ? `0 0 24px color-mix(in oklab, ${it.color} 55%, transparent)` : undefined,
                }}
              >
                <Icon className="h-[18px] w-[18px]" />
              </span>
              {!collapsed && (
                <span className="truncate font-medium">{it.label}</span>
              )}
            </Link>
          );
        })}
      </nav>

      {!collapsed && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-auto glass-card rounded-2xl p-4 text-xs"
        >
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Backend
          </div>
          <div className="mt-1 flex items-center gap-2 text-foreground">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
            FastAPI connected
          </div>
          <div className="mt-1 text-muted-foreground">
            All data streams live from your API.
          </div>
        </motion.div>
      )}
    </motion.aside>
  );
}
