import axios from "axios";

/**
 * Axios client for the EduGuard-AI FastAPI backend.
 * Configure base URL via VITE_API_BASE_URL (e.g. http://localhost:8000).
 */
const baseURL =
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE_URL) ||
  "/api";

export const api = axios.create({
  baseURL,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    // surface a clean error message
    const message =
      err?.response?.data?.detail ||
      err?.response?.data?.message ||
      err?.message ||
      "Request failed";
    return Promise.reject(new Error(message));
  },
);
