/**
 * auth.ts — Google OAuth routes
 *
 * FLOW:
 *   1. GET /auth/login
 *      → redirects browser to Google's OAuth consent screen
 *      → Google asks user to sign in and approve access
 *
 *   2. GET /auth/callback?code=...
 *      → Google redirects back here with an authorization code
 *      → We exchange the code for user identity (via google-auth-library)
 *      → We create a JWT session token
 *      → We redirect to React with the JWT in the URL
 *      → React stores the JWT and uses it for all future requests
 *
 *   3. POST /auth/logout
 *      → Client discards JWT (we don't need server-side session invalidation
 *        for this project — JWT expiry handles it)
 *
 * WHY PKCE ISN'T MANUALLY IMPLEMENTED:
 *   google-auth-library handles the secure OAuth exchange internally.
 *   For a server-side callback flow (code exchange happens on the server,
 *   never in the browser), PKCE is not required — the client secret
 *   provides equivalent protection.
 */

import { Router, Request, Response } from "express";
import { OAuth2Client } from "google-auth-library";
import jwt from "jsonwebtoken";

const router = Router();

const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID!;
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET!;
const JWT_SECRET = process.env.JWT_SECRET!;
const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:3000";
const REDIRECT_URI = `${process.env.EXPRESS_URL || "http://localhost:8000"}/auth/callback`;
const JWT_EXPIRY = "8h"; // users stay logged in for 8 hours

function getOAuthClient(): OAuth2Client {
  return new OAuth2Client(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI);
}

// ---------------------------------------------------------------------------
// GET /auth/login — redirect to Google
// ---------------------------------------------------------------------------
router.get("/login", (_req: Request, res: Response) => {
  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET) {
    res.status(500).json({
      error: "Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env",
    });
    return;
  }

  const client = getOAuthClient();
  const authUrl = client.generateAuthUrl({
    access_type: "offline",
    scope: [
      "https://www.googleapis.com/auth/userinfo.email",
      "https://www.googleapis.com/auth/userinfo.profile",
      "openid",
    ],
    prompt: "select_account", // always show account picker
  });

  res.redirect(authUrl);
});

// ---------------------------------------------------------------------------
// GET /auth/callback — Google redirects here after user approves
// ---------------------------------------------------------------------------
router.get("/callback", async (req: Request, res: Response) => {
  const { code, error } = req.query;

  // User denied access
  if (error) {
    console.error("[auth] OAuth error:", error);
    res.redirect(`${FRONTEND_URL}/login?error=access_denied`);
    return;
  }

  if (!code || typeof code !== "string") {
    res.redirect(`${FRONTEND_URL}/login?error=missing_code`);
    return;
  }

  try {
    const client = getOAuthClient();

    // Exchange authorization code for tokens
    const { tokens } = await client.getToken(code);
    client.setCredentials(tokens);

    // Verify the ID token and extract user info
    const ticket = await client.verifyIdToken({
      idToken: tokens.id_token!,
      audience: GOOGLE_CLIENT_ID,
    });

    const payload = ticket.getPayload();
    if (!payload) {
      throw new Error("Empty token payload from Google");
    }

    const user = {
      sub:     payload.sub,                        // Google user ID
      email:   payload.email || "",
      name:    payload.name || "",
      picture: payload.picture || "",
    };

    console.log(`[auth] Authenticated: ${user.email}`);

    // Create JWT session token
    const sessionToken = jwt.sign(user, JWT_SECRET, { expiresIn: JWT_EXPIRY });

    // Redirect to React with token in URL
    // React will extract it, store in memory, and redirect to /chat
    res.redirect(
      `${FRONTEND_URL}/auth/success?token=${encodeURIComponent(sessionToken)}`
    );

  } catch (err) {
    console.error("[auth] Callback error:", err);
    res.redirect(`${FRONTEND_URL}/login?error=auth_failed`);
  }
});

// ---------------------------------------------------------------------------
// POST /auth/logout — client-side logout (JWT is stateless)
// ---------------------------------------------------------------------------
router.post("/logout", (_req: Request, res: Response) => {
  // JWTs are stateless — we can't invalidate them server-side without a
  // token blacklist (overkill for this project). The client just discards
  // the token. It will expire naturally after 8 hours.
  res.json({ message: "Logged out successfully" });
});

// ---------------------------------------------------------------------------
// GET /auth/me — return current user info (useful for React to verify token)
// ---------------------------------------------------------------------------
router.get("/me", (req: Request, res: Response) => {
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith("Bearer ")) {
    res.status(401).json({ error: "No token provided" });
    return;
  }

  try {
    const token = authHeader.split(" ")[1];
    const user = jwt.verify(token, JWT_SECRET);
    res.json({ user });
  } catch {
    res.status(401).json({ error: "Invalid or expired token" });
  }
});

export default router;