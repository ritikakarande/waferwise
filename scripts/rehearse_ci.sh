#!/bin/bash
# Rehearse the Linux part of .github/workflows/tests.yml in a clean Ubuntu container, before pushing.
#
# Usage (Docker must be running; on Windows use Git Bash):
#     bash scripts/rehearse_ci.sh
#
# The container sees only the files git would push: tracked files plus new files that are not ignored.
# So a file that is missing from the repository, or an install step that is missing from the workflow,
# fails here first. Nothing is left behind: the container is removed when it exits.
set -euo pipefail
cd "$(dirname "$0")/.."

context="$(mktemp -d)"
trap 'rm -rf "$context"' EXIT
git ls-files -co --exclude-standard | tar -czf "$context/project.tgz" -T -

# Docker on Windows needs a Windows-style path for the mounted folder.
mount_dir="$context"
if command -v cygpath > /dev/null; then
    mount_dir="$(cygpath -w "$context")"
fi

# MSYS_NO_PATHCONV stops Git Bash from rewriting the container paths; it does nothing elsewhere.
MSYS_NO_PATHCONV=1 docker run --rm -v "$mount_dir:/context:ro" ubuntu:24.04 bash -c '
    set -euo pipefail
    mkdir /work
    tar -xzf /context/project.tgz -C /work
    cd /work
    bash scripts/ci_linux_steps.sh
'
