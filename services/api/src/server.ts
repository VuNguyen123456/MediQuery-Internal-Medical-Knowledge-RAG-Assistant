/**
 * server.ts — Express API entry point
 *
 * WHY EXPRESS EXISTS:
 *   1. Security proxy — Flask API keys never reach the browser
 *   2. Auth layer — Google OAuth + JWT validation on every request
 *   3. Single public-facing backend — React only talks to port 8000
 *
 * ROUTES:
 *   /auth/*        → Google OAuth flow (login, callback, logout)
 *   /api/query     → proxies to Flask /query (JWT protected)
 *   /api/documents → proxies to Flask /documents (JWT protected)
 *   /health        → health check (public)
 */

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";

// Load .env from project root (two levels up from services/api/src)
dotenv.config({ path: path.resolve(__dirname, "../../../.env") });

import authRoutes from "./routes/auth";
import queryRoutes from "./routes/query";
import { authGuard } from "./middleware/authGuard";

const app = express();
const PORT = process.env.EXPRESS_PORT || 8000;

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.use(express.json());
app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    credentials: true, // allow cookies/auth headers
  })
);

// ---------------------------------------------------------------------------
// Public routes (no auth required)
// ---------------------------------------------------------------------------
app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "mediquery-api" });
});

app.use("/auth", authRoutes);

// ---------------------------------------------------------------------------
// Protected routes (JWT required)
// ---------------------------------------------------------------------------
app.use("/api", authGuard, queryRoutes);

// ---------------------------------------------------------------------------
// 404 handler
// ---------------------------------------------------------------------------
app.use((_req, res) => {
  res.status(404).json({ error: "Route not found" });
});

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`\n MediQuery API running on port ${PORT}`);
  console.log(` Public routes:`);
  console.log(`   GET  /health`);
  console.log(`   GET  /auth/login`);
  console.log(`   GET  /auth/callback`);
  console.log(`   POST /auth/logout`);
  console.log(` Protected routes (JWT required):`);
  console.log(`   POST /api/query`);
  console.log(`   GET  /api/documents`);
  console.log(` Flask service: ${process.env.FLASK_SERVICE_URL || "http://localhost:5000"}\n`);
});

export default app;