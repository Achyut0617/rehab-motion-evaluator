#!/usr/bin/env bash
# Optional: Replace the bundled offline chart renderer with upstream Chart.js 4.4.4 UMD
# when your Jetson Orin Nano has internet access.
set -e
mkdir -p web/static/js
curl -fsSL https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js -o web/static/js/chart.umd.min.js
echo "Downloaded upstream Chart.js to web/static/js/chart.umd.min.js"
