#!/usr/bin/env bash
# Send a notification to the store manager.
#
# Usage:
#   scripts/notify.sh "message"              # text only
#   scripts/notify.sh "caption" photo.jpg    # photo with caption
#
# The channel is chosen from .env in the project root:
#   DISCORD_WEBHOOK_URL=...                    -> Discord (used if set)
#   TELEGRAM_TOKEN=... + TELEGRAM_CHAT_ID=...  -> Telegram
# .env must never be committed (it is listed in .gitignore).
set -euo pipefail

TEXT="${1:?usage: notify.sh \"message\" [photo.jpg]}"
PHOTO="${2:-}"

# Resolve the photo path before changing directory.
if [ -n "$PHOTO" ]; then
  PHOTO="$(realpath "$PHOTO")"
fi

cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
  echo "notify: .env not found in $(pwd)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .env

send_discord() {
  # --form-string sends text literally (no special meaning for @ or <).
  local args=(--form-string "username=Theft Guard" --form-string "content=$TEXT")
  if [ -n "$PHOTO" ]; then
    args+=(-F "file=@$PHOTO")
  fi
  local code
  code="$(curl -s --max-time 15 -o /dev/null -w "%{http_code}" "${args[@]}" "$DISCORD_WEBHOOK_URL")"
  # Discord returns 200 or 204 on success.
  if [ "$code" != "200" ] && [ "$code" != "204" ]; then
    echo "notify: Discord webhook failed (HTTP $code)" >&2
    exit 1
  fi
}

send_telegram() {
  local api="https://api.telegram.org/bot${TELEGRAM_TOKEN}"
  local resp
  if [ -n "$PHOTO" ]; then
    resp="$(curl -s --max-time 15 \
      --form-string chat_id="$TELEGRAM_CHAT_ID" \
      --form-string caption="$TEXT" \
      -F photo=@"$PHOTO" \
      "$api/sendPhoto")"
  else
    resp="$(curl -s --max-time 15 \
      -d chat_id="$TELEGRAM_CHAT_ID" \
      --data-urlencode text="$TEXT" \
      "$api/sendMessage")"
  fi
  if ! grep -q '"ok":true' <<< "$resp"; then
    echo "notify: Telegram API error: $resp" >&2
    exit 1
  fi
}

if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
  send_discord
elif [ -n "${TELEGRAM_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
  send_telegram
else
  echo "notify: set DISCORD_WEBHOOK_URL, or TELEGRAM_TOKEN and TELEGRAM_CHAT_ID, in .env" >&2
  exit 1
fi
