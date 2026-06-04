/**
 * authGuard.ts — JWT validation middleware
 *
 * WHY THIS EXISTS:
 *   Every request to /api/* must include a valid JWT in the Authorization header.
 *   This middleware runs before any protected route handler.
 *   Invalid or missing token → 401, request never reaches Flask.
 *
 * HOW IT WORKS:
 *   1. Extract Bearer token from Authorization header
 *   2. Verify signature using JWT_SECRET
 *   3. Check expiry (jwt.verify handles this automatically)
 *   4. Attach decoded user payload to req for downstream use
 *   5. Call next() to proceed to the route handler
 *
 * USAGE:
 *   app.use("/api", authGuard, queryRoutes);
 *   Any route under /api is automatically protected.
 */

import { Request, Response, NextFunction } from "express";
import jwt from "jsonwebtoken";

// Extend Express Request to carry the decoded user payload
export interface AuthenticatedRequest extends Request {
  user?: {
    email: string;
    name: string;
    picture: string;
    sub: string; // Google user ID
  };
}

export function authGuard(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): void {
  const authHeader = req.headers.authorization;

  // Check Authorization header exists and has Bearer format
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    res.status(401).json({
      error: "Unauthorized",
      message: "Missing or malformed Authorization header. Expected: Bearer <token>",
    });
    return;
  }

  const token = authHeader.split(" ")[1];
  const secret = process.env.JWT_SECRET;

  if (!secret) {
    console.error("[authGuard] JWT_SECRET not set in environment");
    res.status(500).json({ error: "Server misconfiguration" });
    return;
  }

  try {
    // Verify signature + expiry in one call
    const decoded = jwt.verify(token, secret) as AuthenticatedRequest["user"];
    req.user = decoded; // attach to request for downstream handlers
    next();
  } catch (err) {
    if (err instanceof jwt.TokenExpiredError) {
      res.status(401).json({
        error: "Unauthorized",
        message: "Token expired. Please sign in again.",
      });
    } else {
      res.status(401).json({
        error: "Unauthorized",
        message: "Invalid token.",
      });
    }
  }
}