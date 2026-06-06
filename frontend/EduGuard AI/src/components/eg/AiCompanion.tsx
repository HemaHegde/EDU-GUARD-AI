import { motion } from "framer-motion";

/**
 * Soft, friendly AI orb companion: breathing, blinking, glowing.
 * Pass `talking` while a response is streaming.
 */
export function AiCompanion({
  talking = false,
  size = 160,
}: {
  talking?: boolean;
  size?: number;
}) {
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      {/* outer glow */}
      <motion.div
        className="absolute inset-0 rounded-full blur-2xl"
        style={{ background: "radial-gradient(circle, oklch(0.82 0.16 330 / 0.6), transparent 70%)" }}
        animate={{ scale: talking ? [1, 1.15, 1] : [1, 1.05, 1] }}
        transition={{ duration: talking ? 0.9 : 3, repeat: Infinity, ease: "easeInOut" }}
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
        animate={{ y: [0, -6, 0], scale: [1, 1.03, 1] }}
        transition={{ duration: 3.6, repeat: Infinity, ease: "easeInOut" }}
      >
        {/* eyes */}
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
        {/* smile / mouth (animates when talking) */}
        <motion.div
          className="absolute left-1/2 -translate-x-1/2 rounded-full bg-[oklch(0.35_0.07_330)]"
          style={{ bottom: "26%", width: "22%", height: "5%" }}
          animate={talking ? { scaleX: [1, 0.6, 1.1, 0.8, 1], scaleY: [1, 1.6, 0.8, 1.4, 1] } : { scaleY: [1, 1.1, 1] }}
          transition={{ duration: talking ? 0.6 : 3, repeat: Infinity }}
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
    </div>
  );
}
