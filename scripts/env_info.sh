#!/usr/bin/env bash
# Print environment information. Attach this to every benchmark result.
echo "date:      $(date -Iseconds)"
echo "git:       $(git rev-parse --short HEAD 2>/dev/null || echo 'no commit')"
echo "os:        $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "kernel:    $(uname -r)"
echo "cpu:       $(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
echo "threads:   $(nproc)"
echo "ram:       $(free -h | awk '/^Mem:/ {print $2}')"
echo "power:     $(powerprofilesctl get 2>/dev/null || echo unknown)"
echo "g++:       $(g++ --version | head -1)"
echo "cmake:     $(cmake --version | head -1)"
echo "opencv:    $(pkg-config --modversion opencv4 2>/dev/null || echo unknown)"
