#!/bin/bash
# ======================================================================
# P1P Parallel Execution Script
# ======================================================================
# 
# 8 parallel processes (L40S 48GB GPU, 4GB per process)
# Each process runs independently with nohup for stability
# 
# Usage:
#   chmod +x run_parallel.sh
#   ./run_parallel.sh
# 
# Monitoring:
#   tail -f P1P_Project/Logs/P1P_*_p*.log
#   watch -n 1 nvidia-smi
# 
# ======================================================================

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/../P1P_BM_SWOMP_2511v3.py"
LOG_DIR="${SCRIPT_DIR}/Logs"

# --- Setup ---
mkdir -p "$LOG_DIR"

# Timestamp (UTC+9, KST)
TIMESTAMP=$(date -u -d '+9 hours' '+%Y%m%d_%H%M%S' 2>/dev/null || date -v+9H '+%Y%m%d_%H%M%S')

echo "========================================================================"
echo "P1P Parallel Execution (8 partitions)"
echo "========================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Log directory: $LOG_DIR"
echo "Launching 8 parallel processes (4GB per process)..."
echo ""

# --- Launch Processes ---

# Partition 0
LOG_FILE_0="${LOG_DIR}/P1P_${TIMESTAMP}_p0.log"
echo "[Partition 0] Auto-split -> Log: ${LOG_FILE_0}"
nohup python3 -u "$SCRIPT_PATH" --partition 0 > "$LOG_FILE_0" 2>&1 &
PID_0=$!
sleep 2

# Partition 1
LOG_FILE_1="${LOG_DIR}/P1P_${TIMESTAMP}_p1.log"
echo "[Partition 1] Auto-split -> Log: ${LOG_FILE_1}"
nohup python3 -u "$SCRIPT_PATH" --partition 1 > "$LOG_FILE_1" 2>&1 &
PID_1=$!
sleep 2

# Partition 2
LOG_FILE_2="${LOG_DIR}/P1P_${TIMESTAMP}_p2.log"
echo "[Partition 2] Auto-split -> Log: ${LOG_FILE_2}"
nohup python3 -u "$SCRIPT_PATH" --partition 2 > "$LOG_FILE_2" 2>&1 &
PID_2=$!
sleep 2

# Partition 3
LOG_FILE_3="${LOG_DIR}/P1P_${TIMESTAMP}_p3.log"
echo "[Partition 3] Auto-split -> Log: ${LOG_FILE_3}"
nohup python3 -u "$SCRIPT_PATH" --partition 3 > "$LOG_FILE_3" 2>&1 &
PID_3=$!
sleep 2

# Partition 4
LOG_FILE_4="${LOG_DIR}/P1P_${TIMESTAMP}_p4.log"
echo "[Partition 4] Auto-split -> Log: ${LOG_FILE_4}"
nohup python3 -u "$SCRIPT_PATH" --partition 4 > "$LOG_FILE_4" 2>&1 &
PID_4=$!
sleep 2

# Partition 5
LOG_FILE_5="${LOG_DIR}/P1P_${TIMESTAMP}_p5.log"
echo "[Partition 5] Auto-split -> Log: ${LOG_FILE_5}"
nohup python3 -u "$SCRIPT_PATH" --partition 5 > "$LOG_FILE_5" 2>&1 &
PID_5=$!
sleep 2

# Partition 6
LOG_FILE_6="${LOG_DIR}/P1P_${TIMESTAMP}_p6.log"
echo "[Partition 6] Auto-split -> Log: ${LOG_FILE_6}"
nohup python3 -u "$SCRIPT_PATH" --partition 6 > "$LOG_FILE_6" 2>&1 &
PID_6=$!
sleep 2

# Partition 7
LOG_FILE_7="${LOG_DIR}/P1P_${TIMESTAMP}_p7.log"
echo "[Partition 7] Auto-split -> Log: ${LOG_FILE_7}"
nohup python3 -u "$SCRIPT_PATH" --partition 7 > "$LOG_FILE_7" 2>&1 &
PID_7=$!
sleep 2

echo ""
echo "========================================================================"
echo "All 8 partitions launched"
echo "========================================================================"
echo "Process IDs:"
echo "  Partition 0: PID $PID_0"
echo "  Partition 1: PID $PID_1"
echo "  Partition 2: PID $PID_2"
echo "  Partition 3: PID $PID_3"
echo "  Partition 4: PID $PID_4"
echo "  Partition 5: PID $PID_5"
echo "  Partition 6: PID $PID_6"
echo "  Partition 7: PID $PID_7"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/P1P_${TIMESTAMP}_p*.log"
echo "  watch -n 1 nvidia-smi"
echo "========================================================================"

