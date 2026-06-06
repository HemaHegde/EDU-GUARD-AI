import { createFileRoute } from "@tanstack/react-router";

import { motion } from "framer-motion";

import {
  Brain,
  Youtube,
  Upload,
  Timer,
  Activity,
  AlertTriangle,
  BarChart3
} from "lucide-react";

import ReactPlayer from "react-player";

import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis
} from "recharts";

import {
  useCallback,
  useEffect,
  useRef,
  useState
} from "react";

import {
  PageShell,
  ErrorState,
  EmptyState
} from "@/components/eg/PageShell";

// Import your supabase client — adjust path if needed
import { supabase } from "@/lib/supabase";

// =========================
// ROUTE
// =========================

export const Route = createFileRoute(
  "/cognitive"
)({

  head: () => ({

    meta: [

      {
        title:
          "Cognitive Intelligence · EduGuard-AI"
      }

    ]

  }),

  component: CognitivePage,

});

// =========================
// TYPES
// =========================

type LiveMetrics = {

  focus_score: number;

  engagement_score: number;

  confusion_score: number;

  learning_velocity: number;

  ai_risk_probability: number;

  recommendations: string[];

  watch_duration?: number;

  pause_count?: number;

  seek_count?: number;

  cognitive_status?: string;

};

// =========================
// COMPONENT
// =========================

function CognitivePage() {

  // =========================
  // STATES
  // =========================

  const [error, setError] =
    useState("");

  const [videoUrl, setVideoUrl] =
    useState(
      "https://www.youtube.com/watch?v=EAR7De6Goz4"
    );

  const [uploadedVideo, setUploadedVideo] =
    useState("");

  const [pauseCount, setPauseCount] =
    useState(0);

  const [seekCount, setSeekCount] =
    useState(0);

  const [watchDuration, setWatchDuration] =
    useState(0);

  const [sessionEvents, setSessionEvents] =
    useState<string[]>([]);

  const [liveMetrics, setLiveMetrics] =
    useState<LiveMetrics | null>(null);

  const [analyzing, setAnalyzing] =
    useState(false);

  // =========================
  // ROOT CAUSE FIX:
  // Use refs to always hold the latest
  // values of pauseCount, seekCount,
  // and watchDuration.
  //
  // Why: sendTrackingData() is called from
  // onPause/onEnded which are registered as
  // callbacks in ReactPlayer. Without refs,
  // those closures capture the INITIAL state
  // values (0, 0, 0) and never update.
  // Using refs guarantees the fetch always
  // sends the current runtime values.
  // =========================

  const pauseCountRef = useRef(0);
  const seekCountRef = useRef(0);
  const watchDurationRef = useRef(0);
  const lastTrackedSecond = useRef(0);

  // =========================
  // STEP 5: Supabase authenticated user ID
  // localStorage.getItem("user_id") is
  // unreliable — Supabase stores session
  // under its own key, not "user_id".
  // We now fetch the real authenticated user.
  // =========================

  const [userId, setUserId] = useState<string>("");

  useEffect(() => {

    async function resolveUser() {

      try {

        const { data } =
          await supabase.auth.getUser();

        const resolvedId =
          data?.user?.id ||
          localStorage.getItem("user_id") ||
          "";

        setUserId(resolvedId);

        console.log(
          "RESOLVED USER_ID:",
          resolvedId
        );

      } catch (e) {

        console.error(
          "Failed to resolve user:",
          e
        );

        setUserId(
          localStorage.getItem("user_id") || ""
        );

      }

    }

    resolveUser();

  }, []);

  // =========================
  // NORMALIZE URL
  // =========================

  function normalizeYoutubeUrl(
    url: string
  ) {

    if (
      url.includes("youtu.be/")
    ) {

      const videoId =
        url
          .split("youtu.be/")[1]
          ?.split("?")[0];

      return `https://www.youtube.com/watch?v=${videoId}`;

    }

    return url;

  }

  // =========================
  // FINAL VIDEO SOURCE
  // =========================

  const finalVideoSource =
    uploadedVideo || videoUrl;

  // =========================
  // ROOT CAUSE FIX:
  // sendTrackingData wrapped in useCallback
  // with [userId] as dependency.
  //
  // All mutable counters are read from refs
  // (pauseCountRef, seekCountRef,
  // watchDurationRef) — NOT from state.
  // State values are stale inside closures;
  // refs always reflect the latest value.
  // =========================

  const sendTrackingData = useCallback(
    async () => {

      // STEP 1: Debug log before fetch
      console.log("TRACKING DATA", {
        user_id: userId,
        watch_duration: watchDurationRef.current,
        pause_count: pauseCountRef.current,
        seek_count: seekCountRef.current
      });

      setAnalyzing(true);

      try {

        const response =
          await fetch(
            "http://127.0.0.1:8000/cognitive/track",
            {

              method: "POST",

              headers: {

                "Content-Type":
                  "application/json"

              },

              body: JSON.stringify({

                user_id:
                  userId,

                watch_duration:
                  watchDurationRef.current,

                pause_count:
                  pauseCountRef.current,

                seek_count:
                  seekCountRef.current,

                quiz_accuracy:
                  Math.max(
                    40,
                    100 -
                    (
                      pauseCountRef.current * 4 +
                      seekCountRef.current * 3
                    )
                  ),

                response_time:
                  Math.max(
                    5,
                    pauseCountRef.current * 2
                  )

              })

            }
          );

        const data =
          await response.json();

        console.log(
          "AI RESPONSE:",
          data
        );

        setLiveMetrics(data);

        setError("");

      }

      catch (err) {

        console.error(err);

        setError(
          "Backend connection failed"
        );

      }

      finally {

        setAnalyzing(false);

      }

    },
    [userId]
  );

  // =========================
  // STEP 6: Auto track every 20s
  // =========================

  useEffect(() => {

    if (

      watchDuration >=
      lastTrackedSecond.current + 20

    ) {

      lastTrackedSecond.current =
        watchDuration;

      sendTrackingData();

    }

  }, [

    watchDuration,
    sendTrackingData

  ]);

  // =========================
  // METRICS
  // =========================

  const metrics = [

    {
      label: "Focus",
      value:
        liveMetrics?.focus_score || 0,
      color:
        "#60A5FA"
    },

    {
      label: "Engagement",
      value:
        liveMetrics?.engagement_score || 0,
      color:
        "#34D399"
    },

    {
      label: "Confusion",
      value:
        liveMetrics?.confusion_score || 0,
      color:
        "#FB923C"
    },

    {
      label: "Learning Velocity",
      value:
        liveMetrics?.learning_velocity || 0,
      color:
        "#C084FC"
    }

  ];

  // =========================
  // UI
  // =========================

  return (

    <PageShell

      title="Cognitive Intelligence"

      description="Real-time AI cognitive intelligence and behavioral learning analytics."

    >

      {/* VIDEO CARD */}

      <motion.div

        initial={{
          opacity: 0,
          y: 10
        }}

        animate={{
          opacity: 1,
          y: 0
        }}

        className="glass-card rounded-3xl p-6"

      >

        <div className="mb-4 flex items-center gap-2">

          <Youtube className="h-5 w-5 text-primary" />

          <div className="text-lg font-semibold">

            Real Cognitive Video Monitoring

          </div>

        </div>

        {/* URL INPUT */}

        <div className="mb-4">

          <input

            type="text"

            value={videoUrl}

            onChange={(e) =>
              setVideoUrl(
                normalizeYoutubeUrl(
                  e.target.value
                )
              )
            }

            placeholder="Paste YouTube URL"

            className="w-full rounded-2xl border border-border bg-background px-4 py-3 text-sm"

          />

        </div>

        {/* FILE UPLOAD */}

        <div className="mb-6">

          <label className="mb-2 flex items-center gap-2 text-sm font-medium">

            <Upload className="h-4 w-4" />

            Upload Educational Video

          </label>

          <input

            type="file"

            accept="video/*"

            onChange={(e) => {

              const file =
                e.target.files?.[0];

              if (file) {

                const localUrl =
                  URL.createObjectURL(
                    file
                  );

                setUploadedVideo(
                  localUrl
                );

                setError("");

              }

            }}

            className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm"

          />

        </div>

        {/* VIDEO PLAYER */}

        <div className="overflow-hidden rounded-3xl bg-black min-h-[520px]">

          <ReactPlayer

            url={finalVideoSource}

            width="100%"

            height="520px"

            controls={true}

            playing={false}

            muted={false}

            pip={false}

            light={false}

            progressInterval={1000}

            className="react-player"

            config={{

              youtube: {

                playerVars: {

                  modestbranding: 1,
                  rel: 0

                }

              }

            }}

            onReady={() => {

              console.log(
                "Video Ready"
              );

              setError("");

            }}

            onPlay={() => {

              setSessionEvents(
                (prev) => [

                  `▶️ Playback started at ${watchDurationRef.current}s`,
                  ...prev

                ]
              );

            }}

            onPause={() => {

              // STEP 3: Debug pause count
              console.log(
                "PAUSE COUNT:",
                pauseCountRef.current + 1
              );

              // ROOT CAUSE FIX:
              // Update ref FIRST, then state.
              // sendTrackingData reads from ref
              // so it gets the incremented value
              // immediately — not the stale state.
              pauseCountRef.current += 1;

              setPauseCount(
                pauseCountRef.current
              );

              sendTrackingData();

              setSessionEvents(
                (prev) => [

                  `⏸️ Pause at ${watchDurationRef.current}s`,
                  ...prev

                ]
              );

            }}

            onSeek={(seconds) => {

              // STEP 4: Debug seek count
              console.log(
                "SEEK COUNT:",
                seekCountRef.current + 1
              );

              console.log(
                "SEEK DETECTED:",
                seconds
              );

              // ROOT CAUSE FIX:
              // Same pattern — update ref first.
              seekCountRef.current += 1;

              setSeekCount(
                seekCountRef.current
              );

              setSessionEvents(
                (prev) => [

                  `⏩ Seeked to ${Math.floor(seconds)}s`,
                  ...prev

                ]
              );

            }}

            onProgress={(state) => {

              const seconds =
                Math.floor(
                  state.playedSeconds || 0
                );

              // STEP 2: Debug progress
              console.log(
                "PROGRESS:",
                seconds
              );

              // ROOT CAUSE FIX:
              // Keep ref in sync with progress
              // so sendTrackingData always has
              // the latest watchDuration.
              watchDurationRef.current = seconds;

              setWatchDuration(
                seconds
              );

            }}

            onEnded={() => {

              sendTrackingData();

              setSessionEvents(
                (prev) => [

                  "🧠 Final Cognitive Analysis Generated",
                  "✅ Video Completed",
                  ...prev

                ]
              );

            }}

            onError={(e) => {

              console.log(e);

              setError(
                "Video failed to load"
              );

            }}

          />

        </div>

      </motion.div>

      {/* ERROR */}

      {

        error && (

          <div className="mt-6">

            <ErrorState

              message={error}

              onRetry={sendTrackingData}

            />

          </div>

        )

      }

      {/* Analyzing banner */}

      {analyzing && (

        <div className="mt-6 glass-card rounded-3xl p-4 text-sm font-medium text-center">

          Analyzing Cognitive Behaviour...

        </div>

      )}

      {/* LIVE ANALYTICS */}

      <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">

        {

          metrics.map((m, i) => (

            <motion.div

              key={i}

              initial={{
                opacity: 0,
                y: 10
              }}

              animate={{
                opacity: 1,
                y: 0
              }}

              className="glass-card rounded-3xl p-5"

            >

              <div className="text-xs uppercase tracking-wider text-muted-foreground">

                {m.label}

              </div>

              <div className="relative flex items-center justify-center h-[180px]">

                <RadialBarChart

                  width={160}

                  height={160}

                  innerRadius="70%"

                  outerRadius="100%"

                  data={[{

                    value: m.value,
                    fill: m.color

                  }]}

                  startAngle={90}

                  endAngle={-270}

                >

                  <PolarAngleAxis

                    type="number"

                    domain={[0, 100]}

                    tick={false}

                  />

                  <RadialBar

                    dataKey="value"

                    cornerRadius={20}

                  />

                </RadialBarChart>

                <div className="absolute inset-0 flex items-center justify-center">

                  <div className="text-3xl font-bold">

                    {Math.round(
                      m.value
                    )}

                  </div>

                </div>

              </div>

            </motion.div>

          ))

        }

      </div>

      {/* AI RECOMMENDATIONS */}

      <motion.div

        initial={{
          opacity: 0,
          y: 10
        }}

        animate={{
          opacity: 1,
          y: 0
        }}

        className="mt-8 glass-card rounded-3xl p-6"

      >

        <div className="mb-4 flex items-center gap-2">

          <AlertTriangle className="h-4 w-4 text-primary" />

          <div className="text-sm font-semibold">

            Explainable AI Recommendations

          </div>

        </div>

        {

          !liveMetrics?.recommendations?.length ? (

            <EmptyState

              icon={
                <Brain className="h-7 w-7" />
              }

              title="No interventions needed"

              hint="Student cognitive behavior currently stable."

            />

          ) : (

            <div className="space-y-3">

              {

                liveMetrics.recommendations.map(
                  (
                    rec,
                    index
                  ) => (

                    <div

                      key={index}

                      className="rounded-2xl bg-muted/50 p-4 text-sm"

                    >

                      {rec}

                    </div>

                  )
                )

              }

            </div>

          )

        }

      </motion.div>

      {/* TIMELINE */}

      <motion.div

        initial={{
          opacity: 0,
          y: 10
        }}

        animate={{
          opacity: 1,
          y: 0
        }}

        className="mt-8 glass-card rounded-3xl p-6"

      >

        <div className="mb-4 flex items-center gap-2">

          <Activity className="h-4 w-4 text-primary" />

          <div className="text-sm font-semibold">

            Cognitive Session Timeline

          </div>

        </div>

        <div className="space-y-3 max-h-[400px] overflow-y-auto">

          {

            sessionEvents.length === 0 ? (

              <EmptyState

                icon={
                  <Timer className="h-7 w-7" />
                }

                title="No events yet"

                hint="Playback interactions will appear here."

              />

            ) : (

              sessionEvents.map(
                (
                  event,
                  index
                ) => (

                  <div

                    key={index}

                    className="rounded-2xl bg-muted/50 p-4 text-sm"

                  >

                    {event}

                  </div>

                )
              )

            )

          }

        </div>

      </motion.div>

      {/* LIVE ENGINE */}

      <motion.div

        initial={{
          opacity: 0,
          y: 10
        }}

        animate={{
          opacity: 1,
          y: 0
        }}

        className="mt-8 glass-card rounded-3xl p-6"

      >

        <div className="mb-4 flex items-center gap-2">

          <BarChart3 className="h-4 w-4 text-primary" />

          <div className="text-sm font-semibold">

            Live AI Cognitive Engine

          </div>

        </div>

        <div className="grid gap-4 md:grid-cols-5">

          <div className="rounded-2xl bg-muted/50 p-4">

            <div className="text-xs text-muted-foreground">

              Watch Duration

            </div>

            <div className="mt-2 text-2xl font-bold">

              {watchDuration}s

            </div>

          </div>

          <div className="rounded-2xl bg-muted/50 p-4">

            <div className="text-xs text-muted-foreground">

              Pause Count

            </div>

            <div className="mt-2 text-2xl font-bold">

              {pauseCount}

            </div>

          </div>

          <div className="rounded-2xl bg-muted/50 p-4">

            <div className="text-xs text-muted-foreground">

              Replay Events

            </div>

            <div className="mt-2 text-2xl font-bold">

              {seekCount}

            </div>

          </div>

          <div className="rounded-2xl bg-muted/50 p-4">

            <div className="text-xs text-muted-foreground">

              AI Risk Probability

            </div>

            <div className="mt-2 text-2xl font-bold">

              {

                liveMetrics?.ai_risk_probability

                  ?

                  (
                    liveMetrics.ai_risk_probability * 100
                  ).toFixed(1)

                  :

                  "0"

              }%

            </div>

            <div className="mt-2 text-sm">

              {

                liveMetrics?.ai_risk_probability

                ?

                liveMetrics.ai_risk_probability >= 0.75

                  ? "High Risk"

                  :

                  liveMetrics.ai_risk_probability >= 0.45

                  ? "Medium Risk"

                  :

                  "Low Risk"

                :

                "-"

              }

            </div>

          </div>

          <div className="rounded-2xl bg-muted/50 p-4">

            <div className="text-xs text-muted-foreground">

              Cognitive Status

            </div>

            <div className="mt-2 text-lg font-bold">

              {

                liveMetrics?.cognitive_status
                  ?? "Waiting"

              }

            </div>

          </div>

        </div>

      </motion.div>

    </PageShell>

  );

}