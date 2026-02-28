#!/bin/bash
# Test de bloque 0: verificación estructural del repositorio V4
set -e
ERRORS=0

cd "$(dirname "$0")/../.."

echo "=== TEST BLOCK 0: Setup y organización ==="

# 1. Legacy dirs exist
for dir in legacy/v1 legacy/v2 legacy/v3; do
    if [ -d "$dir" ]; then echo "  OK: $dir exists"; else echo "  FAIL: $dir missing"; ERRORS=$((ERRORS+1)); fi
done

# 2. V4 structure exists
for dir in src/v4/common src/v4/pipeline src/v4/evaluation src/v4/adaptation src/v4/ui tests/v4 slurm/jobs/v4 config; do
    if [ -d "$dir" ]; then echo "  OK: $dir exists"; else echo "  FAIL: $dir missing"; ERRORS=$((ERRORS+1)); fi
done

# 3. Config imports
python3 -c "import sys; sys.path.insert(0, '.'); from src.v4.config import MERT_SAMPLE_RATE, REPO_ROOT; assert MERT_SAMPLE_RATE == 24000; print('  OK: config.py imports')" || { echo "  FAIL: config.py"; ERRORS=$((ERRORS+1)); }

# 4. YAML parseable
python3 -c "import yaml; c=yaml.safe_load(open('config/v4.yaml')); assert 'paths' in c; assert 'clustering' in c; print('  OK: v4.yaml parseable')" || { echo "  FAIL: v4.yaml"; ERRORS=$((ERRORS+1)); }

# 5. src/ has no V3 code (only V4)
if [ -d "src/v4" ] && [ ! -d "src/preprocess" ]; then echo "  OK: src/ is clean (V4 only)"; else echo "  FAIL: src/ may still have V3 code"; ERRORS=$((ERRORS+1)); fi

# 6. TODO.md exists
if [ -f "docs/v4/TODO.md" ]; then echo "  OK: TODO.md exists"; else echo "  FAIL: TODO.md missing"; ERRORS=$((ERRORS+1)); fi

# 7. requirements_v4.txt exists
if [ -f "requirements_v4.txt" ]; then echo "  OK: requirements_v4.txt exists"; else echo "  FAIL: requirements_v4.txt missing"; ERRORS=$((ERRORS+1)); fi

echo ""
if [ $ERRORS -eq 0 ]; then echo "BLOCK 0: ALL TESTS PASSED"; else echo "BLOCK 0: $ERRORS TESTS FAILED"; exit 1; fi
