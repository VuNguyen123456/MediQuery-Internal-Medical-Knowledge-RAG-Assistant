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
interface HistoryTurn {
  question: string;
  answer: string;
}

interface PatientContextBody {
  age: string;
  gender: string;
  allergies: string;
  conditions: string;
}

function sanitizeHistory(raw: unknown): HistoryTurn[] {
  if (!Array.isArray(raw)) return [];

  const turns: HistoryTurn[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const { question, answer } = item as Record<string, unknown>;
    if (typeof question !== "string" || typeof answer !== "string") continue;
    const q = question.trim();
    const a = answer.trim();
    if (q && a) turns.push({ question: q, answer: a });
  }

  return turns.slice(-3);
}

function sanitizePatientContext(raw: unknown): PatientContextBody | undefined {
  if (!raw || typeof raw !== "object") return undefined;

  const o = raw as Record<string, unknown>;
  const ctx: PatientContextBody = {
    age: typeof o.age === "string" ? o.age.trim() : "",
    gender: typeof o.gender === "string" ? o.gender.trim() : "",
    allergies: typeof o.allergies === "string" ? o.allergies.trim() : "",
    conditions: typeof o.conditions === "string" ? o.conditions.trim() : "",
  };

  const hasAny = ctx.age || ctx.gender || ctx.allergies || ctx.conditions;
  return hasAny ? ctx : undefined;
}

router.post("/query", async (req: AuthenticatedRequest, res: Response) => {
  const { question, history, patient_context } = req.body;

  if (!question || typeof question !== "string" || !question.trim()) {
    res.status(400).json({ error: "Request body must include a non-empty 'question' field" });
    return;
  }

  const conversationHistory = sanitizeHistory(history);
  const patientContext = sanitizePatientContext(patient_context);

  console.log(
    `[/api/query] User: ${req.user?.email} | Question: ${question.slice(0, 60)}...` +
      (conversationHistory.length ? ` | History: ${conversationHistory.length} turn(s)` : "") +
      (patientContext ? " | Patient context: yes" : "")
  );

  try {
    const flaskBody: {
      question: string;
      history?: HistoryTurn[];
      patient_context?: PatientContextBody;
    } = {
      question: question.trim(),
    };
    if (conversationHistory.length > 0) {
      flaskBody.history = conversationHistory;
    }
    if (patientContext) {
      flaskBody.patient_context = patientContext;
    }

    const flaskResponse = await axios.post(
      `${FLASK_URL}/query`,
      flaskBody,
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