#!/bin/bash
# Submit an LSF job and immediately tail its output log.
# Usage: bash submit.sh <script.bsub>

if [ -z "$1" ]; then
    echo "Usage: bash submit.sh <script.bsub>"
    exit 1
fi

SCRIPT="$1"
if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

# Extract the log pattern from #BSUB -o line
LOG_PATTERN=$(grep '^#BSUB -o' "$SCRIPT" | awk '{print $3}')
if [ -z "$LOG_PATTERN" ]; then
    echo "Error: no #BSUB -o directive found in $SCRIPT"
    exit 1
fi

# Submit and capture job ID
OUTPUT=$(bsub < "$SCRIPT" 2>&1)
echo "$OUTPUT"

JOBID=$(echo "$OUTPUT" | grep -oP '(?<=Job <)\d+')
if [ -z "$JOBID" ]; then
    echo "Error: could not parse job ID from bsub output"
    exit 1
fi

# Build actual log path by replacing %J with job ID
LOGFILE=$(echo "$LOG_PATTERN" | sed "s/%J/$JOBID/")
echo "Tailing $LOGFILE (Ctrl+C to stop, job keeps running)..."
echo ""

# Wait for the log file to appear, then tail it
while [ ! -f "$LOGFILE" ]; do
    sleep 2
done
tail -f "$LOGFILE"
