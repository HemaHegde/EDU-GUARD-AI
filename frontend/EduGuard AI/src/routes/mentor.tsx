import { createFileRoute } from "@tanstack/react-router";

import {
  AnimatePresence,
  motion
} from "framer-motion";

import {
  useEffect,
  useRef,
  useState
} from "react";

import {
  Mic,
  Send,
  Trash2
} from "lucide-react";

import { PageShell } from "@/components/eg/PageShell";
import { AiCompanion } from "@/components/eg/AiCompanion";
import { supabase } from "@/lib/supabase";

// =========================
// ROUTE
// =========================

export const Route = createFileRoute("/mentor")({
  head: () => ({
    meta: [
      {
        title: "AI Mentor · EduGuard-AI"
      }
    ]
  }),
  component: MentorPage,
});

// =========================
// TYPES
// =========================

type Message = {
  role: "user" | "assistant";
  content: string;
};

// =========================
// QUICK SUGGESTION CHIPS
// Improvement 4
// =========================

const SUGGESTION_CHIPS = [
  "Explain Machine Learning",
  "Help me study",
  "I'm stressed",
  "Career advice",
  "Python coding",
];

// =========================
// COMPONENT
// =========================

function MentorPage() {

  const [messages, setMessages] =
    useState<Message[]>([]);

  const [input, setInput] =
    useState("");

  const [sending, setSending] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  // Improvement 1 — proper isSpeaking state
  const [isSpeaking, setIsSpeaking] =
    useState(false);

  // Improvements 5 & 9 — persona/risk/intervention state
  const [persona, setPersona] =
    useState<string | null>(null);

  const [riskLevel, setRiskLevel] =
    useState<string | null>(null);

  const [interventionStyle, setInterventionStyle] =
    useState<string | null>(null);

  const scrollRef =
    useRef<HTMLDivElement>(null);

  const inputRef =
    useRef<HTMLTextAreaElement>(null);

  // =========================
  // AUTO SCROLL
  // =========================

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [messages, sending]);

  // =========================
  // FOCUS INPUT
  // =========================

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // =========================
  // IMPROVEMENT 3 — MEMORY GREETING
  // Fetch history from Supabase, show
  // personalised welcome back message
  // =========================

  useEffect(() => {
    async function loadWelcome() {
      try {
        const {
          data: { user }
        } = await supabase.auth.getUser();

        if (!user) {
          setMessages([{
            role: "assistant",
            content:
              "Hi 👋 I'm Aura. I'm here to help with studies, coding, motivation, career guidance, and everyday questions."
          }]);
          return;
        }

        // Fetch user's first name from profiles table
        const { data: profile } = await supabase
          .from("profiles")
          .select("full_name")
          .eq("id", user.id)
          .single();

        // Fetch their most recent chat topic
        const { data: history } = await supabase
          .from("mentor_history")
          .select("question")
          .eq("user_id", user.id)
          .order("created_at", { ascending: false })
          .limit(1)
          .single();

        const firstName =
          profile?.full_name?.split(" ")[0] ?? null;

        const lastTopic = history?.question ?? null;

        let greeting = "";

        if (firstName && lastTopic) {
          const preview =
            lastTopic.length > 60
              ? lastTopic.slice(0, 60) + "..."
              : lastTopic;
          greeting =
            `Welcome back ${firstName} 🌸\n\nLast time we discussed: "${preview}"\n\nWhat would you like help with today?`;
        } else if (firstName) {
          greeting =
            `Welcome back ${firstName} 🌸\n\nI'm here to help with studies, coding, motivation, career guidance, and everyday questions. What's on your mind?`;
        } else {
          greeting =
            "Hi 👋 I'm Aura. I'm here to help with studies, coding, motivation, career guidance, and everyday questions.";
        }

        setMessages([{
          role: "assistant",
          content: greeting
        }]);

      } catch {
        setMessages([{
          role: "assistant",
          content:
            "Hi 👋 I'm Aura. I'm here to help with studies, coding, motivation, career guidance, and everyday questions."
        }]);
      }
    }

    loadWelcome();
  }, []);

  // =========================
  // IMPROVEMENT 8 — CLEAR CHAT
  // =========================

  function clearChat() {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    setMessages([{
      role: "assistant",
      content: "Hi 👋 I'm Aura. How can I help today?"
    }]);
  }

  // =========================
  // SEND MESSAGE
  // overrideText used by chips
  // =========================

  async function send(overrideText?: string) {

    const text = (overrideText ?? input).trim();

    if (!text || sending) {
      return;
    }

    const userMessage: Message = {
      role: "user",
      content: text
    };

    setMessages((prev) => [
      ...prev,
      userMessage
    ]);

    if (!overrideText) {
      setInput("");
    }

    setSending(true);
    setError(null);

    try {

      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) {
        throw new Error("User not logged in");
      }

      const response = await fetch(
        "http://127.0.0.1:8000/mentor/ask",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            user_id: user.id,
            question: text
          })
        }
      );

      if (!response.ok) {
        throw new Error(
          `API Error ${response.status}`
        );
      }

      const data = await response.json();

      if (data.status === "error") {
        throw new Error(
          data.message || "Mentor failed"
        );
      }

      // Log and save all persona fields
      console.log("Persona:", data.persona);
      console.log("Risk:", data.risk_level);
      console.log("Intervention:", data.intervention_style);

      setPersona(data.persona ?? null);
      setRiskLevel(data.risk_level ?? null);
      setInterventionStyle(data.intervention_style ?? null);

      const mentorReply: Message = {
        role: "assistant",
        content: data.mentor_response
      };

      setMessages((prev) => [
        ...prev,
        mentorReply
      ]);

      // =========================
      // SPEAK RESPONSE
      // Improvement 1 — isSpeaking state
      // =========================

      window.speechSynthesis.cancel();

      const speech =
        new SpeechSynthesisUtterance(
          data.mentor_response
        );

      speech.lang = "en-US";
      speech.rate = 1;
      speech.pitch = 1;
      speech.volume = 1;

      // Improvement 1 — React state tracks speech lifecycle
      speech.onstart = () => setIsSpeaking(true);
      speech.onend = () => setIsSpeaking(false);
      speech.onerror = () => setIsSpeaking(false);

      const voices =
        window.speechSynthesis.getVoices();

      const femaleVoice = voices.find(
        (voice) =>
          voice.name.includes("Female") ||
          voice.name.includes("Zira") ||
          voice.name.includes(
            "Google UK English Female"
          )
      );

      if (femaleVoice) {
        speech.voice = femaleVoice;
      }

      window.speechSynthesis.speak(speech);

    } catch (err: any) {

      setError(
        err.message || "Mentor backend failed"
      );

    } finally {

      setSending(false);

      setTimeout(() => {
        inputRef.current?.focus();
      }, 0);

    }

  }

  // =========================
  // UI
  // =========================

  return (

    <PageShell
      title="AI Mentor"
      description="Your calm, intelligent learning companion — always here, never judging."
    >

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">

        {/* ========================= */}
        {/* LEFT PANEL                */}
        {/* ========================= */}

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card relative flex flex-col items-center gap-4 overflow-hidden rounded-3xl p-6"
        >

          <div
            className="absolute inset-0 -z-10 opacity-70"
            style={{ background: "var(--gradient-soft)" }}
          />

          {/* Improvement 1 — isSpeaking drives animation accurately */}
          <AiCompanion
            talking={sending || isSpeaking}
            size={170}
          />

          <div className="text-center">
            <div className="text-base font-semibold">
              Aura
            </div>
            <div className="text-xs text-muted-foreground">
              Your emotional learning companion
            </div>
          </div>

          {/* ========================= */}
          {/* IMPROVEMENTS 5 & 9        */}
          {/* Positive persona display  */}
          {/* + full AI support card    */}
          {/* ========================= */}

          {(persona || riskLevel || interventionStyle) && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full space-y-2.5 rounded-2xl bg-white/60 p-3.5 text-left"
            >

              {persona && (
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                    🌸 Learning Style
                  </div>
                  <div className="mt-0.5 text-xs font-medium text-foreground">
                    {persona}
                  </div>
                </div>
              )}

              {riskLevel && (
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                    📈 Progress Status
                  </div>
                  <div className="mt-0.5 text-xs font-medium text-foreground">
                    {/* Improvement 5 — friendlier labels */}
                    {riskLevel.toLowerCase() === "high"
                      ? "Needs Support"
                      : riskLevel.toLowerCase() === "medium"
                      ? "On Track"
                      : "Doing Well"}
                  </div>
                </div>
              )}

              {interventionStyle && (
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                    💡 Recommended Support
                  </div>
                  <div className="mt-0.5 text-xs font-medium text-foreground">
                    {interventionStyle}
                  </div>
                </div>
              )}

            </motion.div>
          )}

        </motion.div>

        {/* ========================= */}
        {/* CHAT PANEL                */}
        {/* ========================= */}

        <motion.div
          className="glass-card flex h-[70vh] flex-col overflow-hidden rounded-3xl"
        >

          {/* CHAT HEADER — Improvement 8: Clear Chat button */}
          <div className="flex items-center justify-between border-b border-border/40 px-5 py-3">
            <span className="text-sm font-medium text-foreground/70">
              Chat with Aura
            </span>
            <button
              onClick={clearChat}
              type="button"
              className="flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs text-muted-foreground transition hover:bg-red-50 hover:text-red-500"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear Chat
            </button>
          </div>

          {/* MESSAGES */}
          <div
            ref={scrollRef}
            className="flex-1 space-y-4 overflow-y-auto p-6"
          >

            {messages.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${
                  m.role === "user"
                    ? "justify-end"
                    : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[78%] whitespace-pre-wrap rounded-3xl px-4 py-3 text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-white/85 text-foreground"
                  }`}
                >
                  {m.content}
                </div>
              </motion.div>
            ))}

            {/* Improvement 2 — animated typing indicator */}
            {sending && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="rounded-3xl bg-white/85 px-4 py-3 text-sm">
                  <motion.span
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{
                      repeat: Infinity,
                      duration: 1
                    }}
                  >
                    Aura is typing...
                  </motion.span>
                </div>
              </motion.div>
            )}

          </div>

          {/* ERROR BANNER */}
          {error && (
            <div className="border-t border-red-200 bg-red-50 px-4 py-2 text-xs text-red-600">
              {error}
            </div>
          )}

          {/* ========================= */}
          {/* IMPROVEMENT 4             */}
          {/* Quick suggestion chips    */}
          {/* ========================= */}

          <div className="flex flex-wrap gap-2 px-4 pt-3 pb-1">
            {SUGGESTION_CHIPS.map((chip) => (
              <button
                key={chip}
                type="button"
                onClick={() => send(chip)}
                disabled={sending}
                className="rounded-full border border-border/60 bg-white/70 px-3 py-1 text-xs text-foreground/70 transition hover:bg-primary hover:text-primary-foreground disabled:opacity-40"
              >
                {chip}
              </button>
            ))}
          </div>

          {/* INPUT */}
          <div className="border-t border-border/60 bg-white/40 p-4">

            <div className="flex items-end gap-2 rounded-2xl border border-border/60 bg-white/80 p-2">

              <button
                className="grid h-10 w-10 place-items-center rounded-xl"
                type="button"
              >
                <Mic className="h-[18px] w-[18px]" />
              </button>

              {/* Improvement 7 — Enter sends, Shift+Enter = newline (default textarea behaviour) */}
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) =>
                  setInput(e.target.value)
                }
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                rows={1}
                placeholder="Ask Aura anything… (Shift+Enter for new line)"
                className="flex-1 resize-none bg-transparent px-2 py-2 text-sm outline-none"
              />

              <button
                onClick={() => send()}
                disabled={!input.trim() || sending}
                className="grid h-10 w-10 place-items-center rounded-xl bg-primary text-primary-foreground disabled:opacity-40"
              >
                <Send className="h-[18px] w-[18px]" />
              </button>

            </div>

          </div>

        </motion.div>

      </div>

    </PageShell>

  );

}
