# Rig Assist UI image. Multi-stage: Node builds the React app into static files, then an
# unprivileged nginx serves them and proxies /api to the API. No Node in the final image.
# Images are pinned by tag and digest (a tag can be moved, a digest can't).

# ---- build -------------------------------------------------------------------
FROM node:24.22.1-alpine@sha256:06d7adf3e0ef61a13f1d28682726366f97a15aa1888c934d3ecec86c33918b53 AS build
WORKDIR /ui

# 1. Dependencies only, exactly as package-lock.json says. Reused until the lockfile changes.
COPY package.json package-lock.json .npmrc ./
RUN --mount=type=cache,target=/root/.npm npm ci

# 2. Source → type-check + production build into dist/.
COPY . .
RUN npm run build

# ---- runtime -----------------------------------------------------------------
# nginx stable 1.30.6 (5 CVE fixes over 1.30.5). Runs as uid 101, listens on 8080.
FROM nginxinc/nginx-unprivileged:1.30.6-alpine@sha256:f5116ed4606f3eb015dd4a8e28cb9b351476dc0ad4bc6e9f93b19cec130cdf03

# Where the API is, from inside the network (compose service name; another host on Azure).
# NGINX_ENTRYPOINT_LOCAL_RESOLVERS makes the image read the DNS server from
# /etc/resolv.conf into NGINX_LOCAL_RESOLVERS, which the template's `resolver` uses.
ENV API_UPSTREAM=http://api:8000 \
    NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1

COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /ui/dist /usr/share/nginx/html

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
