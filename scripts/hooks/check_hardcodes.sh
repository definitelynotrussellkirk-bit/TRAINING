#!/bin/bash
#
# Pre-commit hook to prevent hardcoded paths and IPs
# Install: ln -s ../../scripts/hooks/check_hardcodes.sh .git/hooks/pre-commit
#

set -e

# Patterns to check (these should not appear in production code)
# NOTE: These literal strings are for grep detection - not hardcoding!
FORBIDDEN_PATTERNS=(
    "/home/[a-z]*/Desktop/TRAINING"  # User-specific paths
    "192\\.168\\.[0-9]+\\.[0-9]+"    # Private LAN IPs
)

# Files/directories to skip (docs, scratch, config files are OK)
SKIP_PATTERNS=(
    "*.md"
    "scratch/*"
    "config/hosts.json"
    "*.pyc"
    "__pycache__"
    ".git"
    "check_hardcodes.sh"  # This file itself
    "HARDCODE_AUDIT.txt"
    "HARDCODE_FIX_PLAN.md"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

FOUND_VIOLATIONS=0

for file in $STAGED_FILES; do
    # Check if file should be skipped
    SKIP=0
    for pattern in "${SKIP_PATTERNS[@]}"; do
        if [[ "$file" == $pattern ]] || [[ "$file" == *"$pattern"* ]]; then
            SKIP=1
            break
        fi
    done

    if [ $SKIP -eq 1 ]; then
        continue
    fi

    # Only check Python, shell, and HTML files
    if [[ ! "$file" =~ \.(py|sh|html|js)$ ]]; then
        continue
    fi

    # Check each forbidden pattern
    for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
        if grep -q "$pattern" "$file" 2>/dev/null; then
            if [ $FOUND_VIOLATIONS -eq 0 ]; then
                echo -e "${RED}=== HARDCODE VIOLATIONS FOUND ===${NC}"
                echo ""
            fi
            FOUND_VIOLATIONS=1
            echo -e "${YELLOW}File:${NC} $file"
            echo -e "${RED}Contains:${NC} $pattern"
            grep -n "$pattern" "$file" | head -3 | while read line; do
                echo "  $line"
            done
            echo ""
        fi
    done
done

if [ $FOUND_VIOLATIONS -eq 1 ]; then
    echo -e "${RED}=== COMMIT BLOCKED ===${NC}"
    echo ""
    echo "Hardcoded paths/IPs found in staged files."
    echo ""
    echo "Instead of hardcoding, use:"
    echo "  - Python paths:  from core.paths import get_base_dir"
    echo "  - Python hosts:  from core.hosts import get_host, get_service_url"
    echo "  - Shell paths:   \${TRAINING_BASE_DIR:-\$(cd \"\$(dirname \"\$0\")/..\" && pwd)}"
    echo "  - Shell hosts:   \${INFERENCE_HOST:-\$(python3 -c '...')}"
    echo ""
    echo "See: scratch/HARDCODE_FIX_PLAN.md for patterns"
    echo ""
    echo "To bypass (NOT RECOMMENDED): git commit --no-verify"
    exit 1
fi

echo -e "${GREEN}No hardcoded paths/IPs found in staged files.${NC}"
exit 0
