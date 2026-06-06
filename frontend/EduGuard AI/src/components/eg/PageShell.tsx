import { motion } from "framer-motion";
import type { ReactNode } from "react";

export function PageShell({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="mx-auto w-full max-w-7xl space-y-8 p-6 md:p-10"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="text-3xl font-semibold tracking-tight md:text-4xl"
          >
            <span className="text-gradient">{title}</span>
          </motion.h1>
          {description && (
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground md:text-[15px]">
              {description}
            </p>
          )}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </motion.div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl bg-muted/60 ${className}`}
    >
      <div className="absolute inset-0 shimmer-bg" />
    </div>
  );
}

export function EmptyState({
  title,
  hint,
  icon,
}: {
  title: string;
  hint?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="glass-card flex flex-col items-center justify-center gap-3 rounded-3xl p-10 text-center">
      {icon && <div className="text-primary">{icon}</div>}
      <div className="text-base font-medium">{title}</div>
      {hint && <div className="max-w-sm text-sm text-muted-foreground">{hint}</div>}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="glass-card flex flex-col items-center justify-center gap-3 rounded-3xl p-10 text-center">
      <div className="text-base font-medium text-foreground">Couldn't reach the API</div>
      <div className="max-w-md text-sm text-muted-foreground">{message}</div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-[var(--shadow-glow)] transition hover:opacity-90"
        >
          Try again
        </button>
      )}
    </div>
  );
}
