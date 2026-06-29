/**
 * vaccines.ts — Proxy vaccine precaution screening to Flask
 */

import { Router, Response } from "express";
import axios from "axios";
import { AuthenticatedRequest } from "../middleware/authGuard";

const router = Router();

const FLASK_URL = process.env.FLASK_SERVICE_URL || "http://localhost:5000";
const VACCINE_TIMEOUT_MS = Number(process.env.VACCINE_TIMEOUT_MS) || 120000;

router.post("/vaccine-precautions", async (req: AuthenticatedRequest, res: Response) => {
  const { vaccines, condition } = req.body;

  if (!Array.isArray(vaccines) || vaccines.length < 1) {
    res.status(400).json({
      error: "Request body must include 'vaccines' with at least 1 vaccine name",
    });
    return;
  }

  const vaccineList = vaccines
    .filter((v): v is string => typeof v === "string" && v.trim().length > 0)
    .map((v) => v.trim());

  if (vaccineList.length < 1) {
    res.status(400).json({
      error: "At least 1 vaccine name is required",
    });
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
    `[/api/vaccine-precautions] User: ${req.user?.email} | Vaccines: ${vaccineList.join(", ")} | Condition: ${conditionStr}`
  );

  try {
    const flaskResponse = await axios.post(
      `${FLASK_URL}/vaccine-precautions`,
      { vaccines: vaccineList, condition: conditionStr },
      {
        headers: { "Content-Type": "application/json" },
        timeout: VACCINE_TIMEOUT_MS,
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
        res.status(504).json({ error: "Vaccine precaution screening timed out" });
        return;
      }
      if (err.response) {
        res.status(err.response.status).json(err.response.data);
        return;
      }
    }
    console.error("[/api/vaccine-precautions] Unexpected error:", err);
    res.status(500).json({ error: "An unexpected error occurred" });
  }
});

router.get("/vaccines", async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const flaskResponse = await axios.get(`${FLASK_URL}/vaccines`, { timeout: 10000 });
    res.json(flaskResponse.data);
  } catch (err) {
    if (axios.isAxiosError(err) && err.code === "ECONNREFUSED") {
      res.status(503).json({ error: "RAG service unavailable" });
      return;
    }
    res.status(500).json({ error: "Failed to fetch vaccines" });
  }
});

router.get("/vaccines/:vaccineId", async (req: AuthenticatedRequest, res: Response) => {
  const { vaccineId } = req.params;
  try {
    const flaskResponse = await axios.get(
      `${FLASK_URL}/vaccines/${encodeURIComponent(vaccineId)}`,
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
    res.status(500).json({ error: "Failed to fetch vaccine profile" });
  }
});

export default router;
