#!/bin/bash
# Run all analyses
# Usage: bash code/run.sh
# On cluster: set EEG_BASE_DIR env var (done by submit.sh)
# On local Mac: runs with default path in Python code

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="${EEG_BASE_DIR:-$(dirname "$SCRIPT_DIR")}"
cd "$BASE"

PYTHON="${EEG_PYTHON:-$(which python3)}"

echo "Starting at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON"
echo "EEG_BASE_DIR: ${EEG_BASE_DIR:-not set (using default)}"
echo "================================"

# Create output directories
mkdir -p output/correlation output/regression output/rt_prediction

echo ""
echo "[1/2] Analysis 1 & 2: Correlation + Regression"
echo "================================"
$PYTHON -u code/analysis_features.py ${EEG_FLAGS:-} 2>&1 | tee output/run_log_features.txt

echo ""
echo "[2/2] Analysis 3: RT Distribution 3-Way Comparison"
echo "================================"
$PYTHON -u code/analysis_rt.py ${EEG_FLAGS:-} 2>&1 | tee output/run_log_rt.txt

echo ""
echo "[3/3] Summary Reports (EN + KR)"
echo "================================"
$PYTHON -u code/write_summary.py 2>&1

echo ""
echo "================================"
echo "All done at $(date)"
echo ""
echo "Results:"
echo "  output/correlation/correlation_report.md"
echo "  output/regression/regression_report.md"
echo "  output/rt_prediction/rt_comparison_report.md"
echo "  output/summary_EN.md  (presentation)"
echo "  output/summary_KR.md  (understanding)"
