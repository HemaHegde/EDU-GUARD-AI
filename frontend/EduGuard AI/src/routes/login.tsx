import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { Mail, Lock, ArrowRight, Sparkles } from "lucide-react";

import { supabase } from "@/lib/supabase";

export const Route = createFileRoute("/login")({
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);

  async function handleLogin(
    e: React.FormEvent
  ) {
    e.preventDefault();

    setLoading(true);

    const { error } =
      await supabase.auth.signInWithPassword({

        email,
        password,

      });

    setLoading(false);

    if (error) {

      alert(error.message);
      return;

    }

    navigate({
      to: "/",
    });
  }

  return (

    <main className="relative min-h-screen overflow-hidden bg-[var(--gradient-soft)]">

      <div className="grid min-h-screen lg:grid-cols-2">

        {/* LEFT */}

        <section className="hidden lg:flex flex-col justify-between p-14">

          <div className="flex items-center gap-3">

            <div className="grid h-11 w-11 place-items-center rounded-full bg-primary text-white">
              <Sparkles className="h-5 w-5" />
            </div>

            <div>

              <div className="text-xl font-bold">
                EduGuard-AI
              </div>

              <div className="text-sm text-muted-foreground">
                AI Learning Companion
              </div>

            </div>

          </div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
          >

            <div className="text-sm uppercase tracking-[0.3em] text-primary">
              Welcome Back
            </div>

            <h1 className="mt-4 text-6xl font-bold leading-tight">

              Your intelligent learning
              ecosystem is waiting.

            </h1>

            <p className="mt-6 max-w-lg text-lg text-muted-foreground">

              Real-time learning analytics,
              psychological intelligence,
              AI mentorship, and adaptive learning —
              all in one platform.

            </p>

          </motion.div>

          <div className="glass-card rounded-3xl p-6">

            <div className="text-lg font-semibold">
              “EduGuard feels like an intelligent academic companion.”
            </div>

            <div className="mt-2 text-sm text-muted-foreground">
              — Research Faculty Feedback
            </div>

          </div>

        </section>

        {/* RIGHT */}

        <section className="flex items-center justify-center p-6">

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card w-full max-w-md rounded-[2rem] p-10"
          >

            <h2 className="text-4xl font-bold">
              Sign In
            </h2>

            <p className="mt-2 text-muted-foreground">
              Continue your EduGuard-AI journey.
            </p>

            <form
              onSubmit={handleLogin}
              className="mt-8 space-y-5"
            >

              {/* EMAIL */}

              <div>

                <label className="text-sm font-medium">
                  Email
                </label>

                <div className="mt-2 flex items-center rounded-2xl border bg-white/70 px-4">

                  <Mail className="h-4 w-4 text-muted-foreground" />

                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) =>
                      setEmail(e.target.value)
                    }
                    placeholder="you@example.com"
                    className="w-full bg-transparent px-3 py-4 outline-none"
                  />

                </div>

              </div>

              {/* PASSWORD */}

              <div>

                <label className="text-sm font-medium">
                  Password
                </label>

                <div className="mt-2 flex items-center rounded-2xl border bg-white/70 px-4">

                  <Lock className="h-4 w-4 text-muted-foreground" />

                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) =>
                      setPassword(e.target.value)
                    }
                    placeholder="••••••••"
                    className="w-full bg-transparent px-3 py-4 outline-none"
                  />

                </div>

              </div>

              <button
                type="submit"
                disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-full bg-primary py-4 text-white transition hover:scale-[1.01]"
              >

                {loading
                  ? "Signing in..."
                  : "Continue"}

                <ArrowRight className="h-4 w-4" />

              </button>

            </form>

            <p className="mt-8 text-center text-sm text-muted-foreground">

              New here?{" "}

              <Link
                to="/signup"
                className="font-medium text-primary"
              >
                Create account
              </Link>

            </p>

          </motion.div>

        </section>

      </div>

    </main>
  );
}