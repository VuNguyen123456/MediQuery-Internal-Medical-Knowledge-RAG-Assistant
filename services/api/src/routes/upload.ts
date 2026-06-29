/**
 * upload.ts — Proxy routes for PDF upload and delete
 *
 * ROUTES:
 *   POST   /api/upload              → Flask POST /upload
 *   DELETE /api/documents/:filename → Flask DELETE /delete/:filename
 */

import { Router, Response } from "express";
import axios from "axios";
import multer from "multer";
import FormData from "form-data";
import { AuthenticatedRequest } from "../middleware/authGuard";

const router = Router();

const FLASK_URL = process.env.FLASK_SERVICE_URL || "http://localhost:5000";
/** Time to accept file and start job (indexing runs async on Flask). */
const UPLOAD_START_TIMEOUT_MS = Number(process.env.UPLOAD_START_TIMEOUT_MS) || 120000;
const MAX_FILE_SIZE = 50 * 1024 * 1024;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (_req, file, cb) => {
    if (!file.originalname.toLowerCase().endsWith(".pdf")) {
      cb(new Error("Only PDF files are supported"));
      return;
    }
    cb(null, true);
  },
});

function handleMulterError(err: unknown, res: Response): boolean {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      res.status(413).json({ error: "File exceeds 50MB limit" });
      return true;
    }
    res.status(400).json({ error: err.message });
    return true;
  }
  if (err instanceof Error && err.message === "Only PDF files are supported") {
    res.status(400).json({ error: err.message });
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// POST /api/upload — ingest a PDF into the knowledge base
// ---------------------------------------------------------------------------
router.post("/upload", (req: AuthenticatedRequest, res: Response) => {
  upload.fields([{ name: "file", maxCount: 1 }])(req, res, async (err: unknown) => {
    if (err && handleMulterError(err, res)) return;

    const files = req.files as { file?: Express.Multer.File[] } | undefined;
    const file = files?.file?.[0];

    if (!file) {
      res.status(400).json({ error: "Request must include a PDF file" });
      return;
    }

    const section =
      typeof req.body?.section === "string" ? req.body.section.trim() : "";

    console.log(
      `[/api/upload] User: ${req.user?.email} | File: ${file.originalname} (${file.size} bytes) | Section: ${section || "(none)"}`
    );

    try {
      const form = new FormData();
      form.append("file", file.buffer, {
        filename: file.originalname,
        contentType: "application/pdf",
      });
      if (section) {
        form.append("section", section);
      }

      const flaskResponse = await axios.post(`${FLASK_URL}/upload`, form, {
        headers: form.getHeaders(),
        timeout: UPLOAD_START_TIMEOUT_MS,
        maxContentLength: MAX_FILE_SIZE,
        maxBodyLength: MAX_FILE_SIZE,
        validateStatus: (status) => status === 202 || status < 300,
      });

      res.status(flaskResponse.status).json(flaskResponse.data);
    } catch (proxyErr) {
      if (axios.isAxiosError(proxyErr)) {
        if (proxyErr.code === "ECONNREFUSED") {
          res.status(503).json({ error: "RAG service unavailable. Is Flask running on port 5000?" });
          return;
        }
        if (proxyErr.code === "ECONNABORTED") {
          res.status(504).json({ error: "Upload timed out — try a smaller PDF or try again." });
          return;
        }
        if (proxyErr.response) {
          res.status(proxyErr.response.status).json(proxyErr.response.data);
          return;
        }
      }
      console.error("[/api/upload] Unexpected error:", proxyErr);
      res.status(500).json({ error: "An unexpected error occurred during upload" });
    }
  });
});

// ---------------------------------------------------------------------------
// GET /api/upload/status/:jobId — poll ingestion progress
// ---------------------------------------------------------------------------
router.get(
  "/upload/status/:jobId",
  async (req: AuthenticatedRequest, res: Response) => {
    const { jobId } = req.params;
    if (!jobId) {
      res.status(400).json({ error: "Job ID is required" });
      return;
    }

    try {
      const flaskResponse = await axios.get(
        `${FLASK_URL}/upload/status/${encodeURIComponent(jobId)}`,
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
      console.error("[/api/upload/status] Unexpected error:", err);
      res.status(500).json({ error: "Failed to fetch upload status" });
    }
  }
);

// ---------------------------------------------------------------------------
// DELETE /api/documents?path=section/file.pdf — remove PDF + vectors
// ---------------------------------------------------------------------------
router.delete("/documents", async (req: AuthenticatedRequest, res: Response) => {
  const docPath = typeof req.query.path === "string" ? req.query.path.trim() : "";

  if (!docPath || !docPath.toLowerCase().endsWith(".pdf") || docPath.includes("..")) {
    res.status(400).json({ error: "Invalid document path" });
    return;
  }

  console.log(`[/api/documents] User: ${req.user?.email} | Delete: ${docPath}`);

  try {
    const flaskResponse = await axios.delete(
      `${FLASK_URL}/delete/${encodeURIComponent(docPath)}`,
      { timeout: 30000 }
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
    console.error("[/api/documents] Unexpected error:", err);
    res.status(500).json({ error: "Failed to delete document" });
  }
});

export default router;
