#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root_dir:$root_dir/../deer-flow/backend/packages/extension-api${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'
cd "$root_dir"
exec python3 server.py "$@"
