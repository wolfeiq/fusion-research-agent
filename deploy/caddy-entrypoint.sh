#!/bin/sh
set -eu
FUSION_PROXY_TOKEN="$(cat /run/secrets/proxy_token)"
export FUSION_PROXY_TOKEN
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
