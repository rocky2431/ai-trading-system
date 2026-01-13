# ==================== Builder Stage ====================
FROM node:25-alpine as builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build arguments
ARG VITE_API_URL=http://localhost:8000

# Build application
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ==================== Production Stage ====================
FROM nginx:alpine as production

# Install envsubst (gettext package)
RUN apk add --no-cache gettext

# Copy nginx configuration template (for Railway dynamic PORT)
COPY nginx.conf.template /etc/nginx/conf.d/default.conf.template
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Create non-root user with write permissions
RUN addgroup -g 1000 iqfmp && \
    adduser -u 1000 -G iqfmp -s /bin/sh -D iqfmp && \
    chown -R iqfmp:iqfmp /usr/share/nginx/html && \
    chown -R iqfmp:iqfmp /var/cache/nginx && \
    chown -R iqfmp:iqfmp /var/log/nginx && \
    chown -R iqfmp:iqfmp /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && \
    chown -R iqfmp:iqfmp /var/run/nginx.pid

# Switch to non-root user
USER iqfmp

# Health check (uses PORT env var, defaults to 80)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT:-80}/ || exit 1

# Default port (Railway will override via PORT env)
ENV PORT=80
EXPOSE ${PORT}

# Start with dynamic port substitution
CMD ["/start.sh"]
