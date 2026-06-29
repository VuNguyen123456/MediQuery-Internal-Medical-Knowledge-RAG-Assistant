/**
 * interactions.ts — Proxy drug interaction screening to Flask
 */

import { Router, Response } from "express";
import axios from "axios";
import { AuthenticatedRequest } from "../middleware/authGuard";

const router = Router();

const FLASK_URL = process.env.FLASK_SERVICE_URL || "http://localhost:5000";
const INTERACTION_TIMEOUT_MS = Number(process.env.INTERACTION_TIMEOUT_MS) || 120000;

router.post("/interactions", async (req: AuthenticatedRequest, res: Response) => {
  const { drugs, condition } = req.body;

  if (!Array.isArray(drugs) || drugs.length < 2) {
    res.status(400).json({
      error: "Request body must include 'drugs' with at least 2 drug names",
    });
    return;
  }

  const drugList = drugs
    .filter((d): d is string => typeof d === "string" && d.trim().length > 0)
    .map((d) => d.trim());

  if (drugList.length < 2) {
    res.status(400).json({
      error: "At least 2 distinct drug names are required",
    });
    return;
  }

  console.log(
    `[/api/interactions] User: ${req.user?.email} | Drugs: ${drugList.join(", ")}`
  );

  try {
    const body: { drugs: string[]; condition?: string } = { drugs: drugList };
    if (typeof condition === "string" && condition.trim()) {
      body.condition = condition.trim();
    }

    const flaskResponse = await axios.post(`${FLASK_URL}/interactions`, body, {
      headers: { "Content-Type": "application/json" },
      timeout: INTERACTION_TIMEOUT_MS,
    });

    res.json(flaskResponse.data);
  } catch (err) {
    if (axios.isAxiosError(err)) {
      if (err.code === "ECONNREFUSED") {
        res.status(503).json({ error: "RAG service unavailable" });
        return;
      }
      if (err.code === "ECONNABORTED") {
        res.status(504).json({ error: "Interaction screening timed out" });
        return;
      }
      if (err.response) {
        res.status(err.response.status).json(err.response.data);
        return;
      }
    }
    console.error("[/api/interactions] Unexpected error:", err);
    res.status(500).json({ error: "An unexpected error occurred" });
  }
});

router.get("/drugs", async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const flaskResponse = await axios.get(`${FLASK_URL}/drugs`, { timeout: 10000 });
    res.json(flaskResponse.data);
  } catch (err) {
    if (axios.isAxiosError(err) && err.code === "ECONNREFUSED") {
      res.status(503).json({ error: "RAG service unavailable" });
      return;
    }
    res.status(500).json({ error: "Failed to fetch drugs" });
  }
});

router.get("/drugs/:drugId", async (req: AuthenticatedRequest, res: Response) => {
  const { drugId } = req.params;
  try {
    const flaskResponse = await axios.get(
      `${FLASK_URL}/drugs/${encodeURIComponent(drugId)}`,
      { timeout: 10000 }
    );
    res.json(flaskResponse.data);
  } catch (err) {
    if (axios.isAxiosError(err)) {
      if (err.code === "ECONNREFUSED") {
        res.status(503).json({ error: "RAG service unavailable" });
        return;
      }
      if (err.response) {
        res.status(err.response.status).json(err.response.data);
        return;
      }
    }
    res.status(500).json({ error: "Failed to fetch drug profile" });
  }
});

router.post("/drug-precautions", async (req: AuthenticatedRequest, res: Response) => {
  const { drugs, condition } = req.body;

  if (!Array.isArray(drugs) || drugs.length < 1) {
    res.status(400).json({
      error: "Request body must include 'drugs' with at least 1 drug name",
    });
    return;
  }

  const drugList = drugs
    .filter((d): d is string => typeof d === "string" && d.trim().length > 0)
    .map((d) => d.trim());

  if (drugList.length < 1) {
    res.status(400).json({ error: "At least 1 drug name is required" });
    return;
  }

  const conditionStr =
    typeof condition === "string" && condition.trim() ? condition.trim() : "";

  if (!conditionStr) {
    res.status(400).json({
      error: "Request body must include a non-empty 'condition' (patient context)",
    });
    return;
  }

  console.log(
    `[/api/drug-precautions] User: ${req.user?.email} | Drugs: ${drugList.join(", ")} | Condition: ${conditionStr}`
  );

  try {
    const flaskResponse = await axios.post(
      `${FLASK_URL}/drug-precautions`,
      { drugs: drugList, condition: conditionStr },
      {
        headers: { "Content-Type": "application/json" },
        timeout: INTERACTION_TIMEOUT_MS,
      }
    );
    res.json(flaskResponse.data);
  } catch (err) {
    if (axios.isAxiosError(err)) {
      if (err.code === "ECONNREFUSED") {
        res.status(503).json({ error: "RAG service unavailable" });
        return;
      }
      if (err.code === "ECONNABORTED") {
        res.status(504).json({ error: "Drug precaution screening timed out" });
        return;
      }
      if (err.response) {
        const data = err.response.data;
        if (data && typeof data === "object" && "error" in data) {
          res.status(err.response.status).json(data);
          return;
        }
        if (err.response.status === 404) {
          res.status(404).json({
            error:
              "Drug precaution endpoint not found on RAG service — restart Flask (python app.py) and Express.",
          });
          return;
        }
        res.status(err.response.status).json({
          error: "Drug precaution screening failed on RAG service",
        });
        return;
      }
    }
    console.error("[/api/drug-precautions] Unexpected error:", err);
    res.status(500).json({ error: "An unexpected error occurred" });
  }
});

export default router;
