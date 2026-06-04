# Express API Service — Node 20 slim (multi-stage build)
FROM node:20-slim AS builder

WORKDIR /app

# Full install — devDependencies include @types/* and typescript needed for tsc
COPY services/api/package*.json ./
RUN npm ci

COPY services/api/tsconfig.json ./
COPY services/api/src/ ./src/
RUN npm run build

# Production image — runtime deps only
FROM node:20-slim

WORKDIR /app

COPY services/api/package*.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/dist ./dist

EXPOSE 8005

CMD ["node", "dist/server.js"]
