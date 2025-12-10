# ==================== Builder Stage ====================
FROM node:20-alpine as builder

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

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Create non-root user
RUN addgroup -g 1000 iqfmp && \
    adduser -u 1000 -G iqfmp -s /bin/sh -D iqfmp && \
    chown -R iqfmp:iqfmp /usr/share/nginx/html && \
    chown -R iqfmp:iqfmp /var/cache/nginx && \
    chown -R iqfmp:iqfmp /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R iqfmp:iqfmp /var/run/nginx.pid

# Switch to non-root user
USER iqfmp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
