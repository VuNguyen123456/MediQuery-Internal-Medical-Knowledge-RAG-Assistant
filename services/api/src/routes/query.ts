/**
 * query.ts — Proxy routes to the Flask RAG service
 *
 * WHY A PROXY:
 *   React can't call Flask directly because:
 *   1. Flask runs on an internal Docker network (not exposed to browser)
 *   2. Even locally, Flask keys should never be browser-accessible
 *   3. Auth validation must happen before RAG queries are processed
 *
 *   Express receives the request, validates JWT (via authGuard middleware
 *   applied in server.ts), then forwards to Flask internally.
 *
 * ROUTES:
 *   POST /api/query      → Flask POST /query
 *   GET  /api/documents  → Flask GET /documents
 */

import { Router, Response } from "express";
import axios from "axios";
import { AuthenticatedRequest } from "../middleware/authGuard";

const router = Router();

const FLASK_URL = process.env.FLASK_SERVICE_URL || "http://localhost:5000";
const RAG_QUERY_TIMEOUT_MS = Number(process.env.RAG_QUERY_TIMEOUT_MS) || 120000;

// ---------------------------------------------------------------------------
// POST /api/query — run RAG pipeline
// ---------------------------------------------------------------------------
router.post("/query", async (req: AuthenticatedRequest, res: Response) => {
  const { question } = req.body;

  if (!question || typeof question !== "string" || !question.trim()) {
    res.status(400).json({ error: "Request body must include a non-empty 'question' field" });
    return;
  }

  console.log(`[/api/query] User: ${req.user?.email} | Question: ${question.slice(0, 60)}...`);

  try {
    const flaskResponse = await axios.post(
      `${FLASK_URL}/query`,
      { question: question.trim() },
      {
        headers: { "Content-Type": "application/json" },
        timeout: RAG_QUERY_TIMEOUT_MS,
      }
    );

    res.json(flaskResponse.data);

  } catch (err) {
    if (axios.isAxiosError(err)) {
      if (err.code === "ECONNREFUSED") {
        console.error("[/api/query] Flask service unreachable");
        res.status(503).json({
          error: "RAG service unavailable. Is Flask running on port 5000?",
        });
        return;
      }
      if (err.code === "ECONNABORTED") {
        console.error("[/api/query] Flask request timed out");
        res.status(504).json({
          error:
            "RAG query timed out. First query after restart can be slow — try again.",
        });
        return;
      }
      if (err.response) {
        // Flask returned an error — forward it
        res.status(err.response.status).json(err.response.data);
        return;
      }
    }
    console.error("[/api/query] Unexpected error:", err);
    res.status(500).json({ error: "An unexpected error occurred" });
  }
});

// ---------------------------------------------------------------------------
// GET /api/documents — list indexed documents
// ---------------------------------------------------------------------------
router.get("/documents", async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const flaskResponse = await axios.get(`${FLASK_URL}/documents`, {
      timeout: 5000,
    });
    res.json(flaskResponse.data);
  } catch (err) {
    if (axios.isAxiosError(err) && err.code === "ECONNREFUSED") {
      res.status(503).json({ error: "RAG service unavailable" });
      return;
    }
    res.status(500).json({ error: "Failed to fetch documents" });
  }
});

export default router;