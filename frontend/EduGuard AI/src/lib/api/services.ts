import { api } from "./client";

/**
 * Service layer mapping to the EduGuard-AI FastAPI backend.
 * Endpoints are placeholders aligned to the product spec — adjust paths
 * to match the backend's actual routes when wiring up.
 */

// --- Dashboard / Overview ---
export const dashboardService = {
  getOverview: () => api.get("/dashboard/overview").then((r) => r.data),
  getEngagementTrend: () => api.get("/dashboard/engagement-trend").then((r) => r.data),
  getPersonaDistribution: () => api.get("/dashboard/persona-distribution").then((r) => r.data),
};

// --- Academic Risk ---
export const riskService = {
  list: () => api.get("/risk/students").then((r) => r.data),
  getStudent: (id: string) => api.get(`/risk/students/${id}`).then((r) => r.data),
  getExplanations: (id: string) => api.get(`/risk/students/${id}/explain`).then((r) => r.data),
};

// --- Cognitive Intelligence ---
export const cognitiveService = {
  getMetrics: (studentId?: string) =>
    api
      .get("/cognitive/metrics", { params: studentId ? { student_id: studentId } : undefined })
      .then((r) => r.data),
  getInsights: () => api.get("/cognitive/insights").then((r) => r.data),
};

// --- Persona Intelligence ---
export const personaService = {
  list: () => api.get("/personas").then((r) => r.data),
  getDistribution: () => api.get("/personas/distribution").then((r) => r.data),
};

// --- AI Mentor ---
export type MentorMessage = { role: "user" | "assistant"; content: string };
export const mentorService = {
  send: (payload: { message: string; history: MentorMessage[]; session_id?: string }) =>
    api.post("/mentor/chat", payload).then((r) => r.data as { reply: string; session_id?: string }),
};

// --- Quiz + Flashcards ---
export const learnService = {
  getQuiz: (topic?: string) =>
    api.get("/learn/quiz", { params: topic ? { topic } : undefined }).then((r) => r.data),
  submitQuiz: (payload: { quiz_id: string; answers: Record<string, string> }) =>
    api.post("/learn/quiz/submit", payload).then((r) => r.data),
  getFlashcards: (topic?: string) =>
    api.get("/learn/flashcards", { params: topic ? { topic } : undefined }).then((r) => r.data),
  getNotes: (topic?: string) =>
    api.get("/learn/notes", { params: topic ? { topic } : undefined }).then((r) => r.data),
};

// --- Video Intelligence ---
export const videoService = {
  process: (payload: { url?: string; file_id?: string }) =>
    api.post("/video/process", payload).then((r) => r.data),
  status: (jobId: string) => api.get(`/video/status/${jobId}`).then((r) => r.data),
  getTranscript: (jobId: string) => api.get(`/video/${jobId}/transcript`).then((r) => r.data),
  getEmbeddings: (jobId: string) => api.get(`/video/${jobId}/embeddings`).then((r) => r.data),
  getGeneratedQuiz: (jobId: string) => api.get(`/video/${jobId}/quiz`).then((r) => r.data),
  getGeneratedFlashcards: (jobId: string) => api.get(`/video/${jobId}/flashcards`).then((r) => r.data),
};

// --- Research Analytics ---
export const researchService = {
  getOverview: () => api.get("/research/overview").then((r) => r.data),
  getModelMetrics: () => api.get("/research/model-metrics").then((r) => r.data),
  getCorrelations: () => api.get("/research/correlations").then((r) => r.data),
  getShapImportance: () => api.get("/research/shap-importance").then((r) => r.data),
  getPersonaProfiles: () => api.get("/research/persona-profiles").then((r) => r.data),

  getClusterAnova: () => api.get("/research/cluster-anova").then((r) => r.data),
};
