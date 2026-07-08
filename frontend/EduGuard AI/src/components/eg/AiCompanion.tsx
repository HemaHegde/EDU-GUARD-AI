import { motion } from "framer-motion";

/**
 * Soft, friendly AI orb companion: breathing, blinking, glowing.
 *
 * Drive its behaviour with `state`:
 *  - "greeting"  one-time wave + smile, played once on mount / handoff
 *  - "idle"      slow breathing, floating, soft glow (default)
 *  - "thinking"  brighter/expanding glow, slow rotation, dots below
 *  - "speaking"  faster mouth animation, gentle pulse, brighter glow
 *  - "listening" very subtle bounce, glow slightly brighter than idle
 */

type AuraState =
  | "greeting"
  | "idle"
  | "thinking"
  | "speaking"
  | "listening";

// =========================
// OUTER GLOW — per-state animation
// =========================

const outerGlowAnimation: Record<AuraState, any> = {
  greeting: {
    scale: [1, 1.12, 1],
    opacity: [0.6, 0.85, 0.6],
  },
  idle: {
    scale: [1, 1.08, 1],
    opacity: [0.6, 0.7, 0.6],
  },
  thinking: {
    scale: [1, 1.3, 1.18, 1.3, 1],
    opacity: [0.6, 0.95, 0.8, 0.95, 0.6],
  },
  speaking: {
    scale: [1, 1.18, 1.05, 1.18, 1],
    opacity: [0.7, 1, 0.8, 1, 0.7],
  },
  listening: {
    scale: [1, 1.06, 1],
    opacity: [0.65, 0.78, 0.65],
  },
};

const outerGlowTransition: Record<AuraState, any> = {
  greeting: { duration: 1.2, ease: "easeInOut" },
  idle: { duration: 3.5, repeat: Infinity, ease: "easeInOut" },
  thinking: { duration: 1.8, repeat: Infinity, ease: "easeInOut" },
  speaking: { duration: 0.7, repeat: Infinity, ease: "easeInOut" },
  listening: { duration: 1.8, repeat: Infinity, ease: "easeInOut" },
};

// =========================
// ORB — per-state animation
// =========================

const orbAnimation: Record<AuraState, any> = {
  greeting: {
    rotate: [0, 10, -10, 0],
    y: [0, -8, 0],
    scale: [1, 1.05, 1],
  },
  idle: {
    rotate: 0,
    y: [0, -5, 0],
    scale: [1, 1.03, 1],
  },
  thinking: {
    rotate: [-3, 3, -3],
    y: [0, -5, 0],
    scale: [1, 1.04, 1],
  },
  speaking: {
    rotate: 0,
    y: [0, -5, 0],
    scale: [1, 1.06, 1, 1.04, 1],
  },
  listening: {
    rotate: 0,
    y: [0, -3, 0],
    scale: [1, 1.01, 1],
  },
};

const orbTransition: Record<AuraState, any> = {
  greeting: { duration: 1.2, ease: "easeInOut" },
  idle: { duration: 3.5, repeat: Infinity, ease: "easeInOut" },
  thinking: { duration: 2.5, repeat: Infinity, ease: "easeInOut" },
  speaking: { duration: 0.6, repeat: Infinity, ease: "easeInOut" },
  listening: { duration: 1.8, repeat: Infinity, ease: "easeInOut" },
};

// =========================
// MOUTH — only animates while speaking
// =========================

const mouthAnimation = {
  scaleX: [1, 0.6, 1.1, 0.75, 1],
  scaleY: [1, 1.7, 0.8, 1.5, 1],
};

const mouthTransition = {
  duration: 0.45,
  repeat: Infinity,
  ease: "easeInOut",
};

const mouthRelaxed = {
  scaleX: 1,
  scaleY: 1,
};

// Greeting gets a one-time gentle smile lift, everything else
// (besides speaking) stays relaxed.
const mouthGreetingAnimation = {
  scaleY: [1, 1.3, 1],
};

const mouthGreetingTransition = {
  duration: 1.2,
  ease: "easeInOut",
};

export function AiCompanion({
  state = "idle",
  size = 160,
}: {
  state?: AuraState;
  size?: number;
}) {
  const isThinking = state === "thinking";
  const isSpeaking = state === "speaking";
  const isGreeting = state === "greeting";

  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      {/* outer glow */}
      <motion.div
        className="absolute inset-0 rounded-full blur-2xl"
        style={{ background: "radial-gradient(circle, oklch(0.82 0.16 330 / 0.6), transparent 70%)" }}
        animate={outerGlowAnimation[state]}
        transition={outerGlowTransition[state]}
      />

      {/* orb */}
      <motion.div
        className="relative rounded-full"
        style={{
          width: size * 0.7,
          height: size * 0.7,
          background:
            "radial-gradient(circle at 30% 28%, oklch(0.98 0.04 320), oklch(0.82 0.12 330) 55%, oklch(0.7 0.14 290) 100%)",
          boxShadow:
            "inset 0 -14px 30px oklch(0.55 0.15 290 / 0.5), inset 0 14px 24px oklch(1 0 0 / 0.6), 0 20px 60px oklch(0.7 0.15 330 / 0.45)",
        }}
        animate={orbAnimation[state]}
        transition={orbTransition[state]}
      >
        {/* eyes — blink continues in every state */}
        <div className="absolute inset-0 flex items-center justify-center gap-[14%]" style={{ top: "8%" }}>
          {[0, 1].map((i) => (
            <motion.span
              key={i}
              className="block rounded-full bg-[oklch(0.25_0.05_290)]"
              style={{ width: "10%", height: "16%" }}
              animate={{ scaleY: [1, 1, 0.1, 1, 1] }}
              transition={{ duration: 5, repeat: Infinity, times: [0, 0.92, 0.95, 0.98, 1], ease: "easeInOut" }}
            />
          ))}
        </div>

        {/* smile / mouth — only animates while speaking */}
        <motion.div
          className="absolute left-1/2 -translate-x-1/2 rounded-full bg-[oklch(0.35_0.07_330)]"
          style={{ bottom: "26%", width: "22%", height: "5%" }}
          animate={
            isSpeaking
              ? mouthAnimation
              : isGreeting
              ? mouthGreetingAnimation
              : mouthRelaxed
          }
          transition={
            isSpeaking
              ? mouthTransition
              : isGreeting
              ? mouthGreetingTransition
              : { duration: 0.3 }
          }
        />

        {/* cheek blush */}
        <span
          className="absolute rounded-full opacity-70 blur-md"
          style={{ width: "14%", height: "10%", background: "oklch(0.82 0.14 10)", left: "16%", top: "55%" }}
        />
        <span
          className="absolute rounded-full opacity-70 blur-md"
          style={{ width: "14%", height: "10%", background: "oklch(0.82 0.14 10)", right: "16%", top: "55%" }}
        />
      </motion.div>

      {/* thinking dots — only visible during "thinking" */}
      {isThinking && (
        <div
          className="absolute flex items-center gap-1.5"
          style={{ top: "100%", marginTop: 10 }}
        >
          {[0, 1, 2].map((i) => (
            <motion.span
              key={i}
              className="block rounded-full bg-[oklch(0.7_0.14_290)]"
              style={{ width: size * 0.045, height: size * 0.045 }}
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{
                duration: 1.1,
                repeat: Infinity,
                ease: "easeInOut",
                delay: i * 0.2,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
