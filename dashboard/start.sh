#!/bin/sh
# Railway startup script - substitutes PORT env var into nginx config

# Default to port 80 if PORT not set
export PORT=${PORT:-80}

# Substitute environment variables in nginx config
envsubst '${PORT}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

# Start nginx
exec nginx -g 'daemon off;'
