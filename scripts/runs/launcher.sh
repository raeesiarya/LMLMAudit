#!/bin/bash
set -euo pipefail

ACCOUNT=fc_cosi
WALL_MIN=06:00:00
WALL=08:00:00
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

# CPU rules
CPUS_A5000=4
CPUS_L40=8
CPUS_A40=8
CPUS_2080TI=2
CPUS_1080TI=2

declare -a JOBS=()

##############################################
# submit(partition, qos, gres_string, gpus, cpus_per_task)
##############################################
submit() {
    local partition="$1"
    local qos="$2"
    local gres="$3"
    local gpus="$4"
    local cpt="$5"

    sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$partition" \
        --qos="$qos" \
        --time-min="$WALL_MIN" \
        --time="$WALL" \
        --nodes=1 \
        --gres="$gres" \
        --ntasks-per-node="$gpus" \
        --cpus-per-task="$cpt" \
        --output="${LOG_DIR}/slurm-%j.out" \
        --error="${LOG_DIR}/slurm-%j.err" \
        payload.sh
}

echo "Submitting multi-partition GPU candidates..."

# savio4_gpu A5000
for g in 1; do
    jid=$(submit "savio4_gpu" "a5k_gpu4_normal" "gpu:A5000:${g}" "$g" "$CPUS_A5000")
    echo "savio4_gpu A5000 (${g} GPU) -> $jid"
    JOBS+=("$jid")
done

# savio4_gpu L40
for g in 1; do
    jid=$(submit "savio4_gpu" "savio_lowprio" "gpu:L40:${g}" "$g" "$CPUS_L40")
    echo "savio4_gpu L40 (${g} GPU lowprio) -> $jid"
    JOBS+=("$jid")
done

# savio3_gpu A40
for g in 1; do
    jid=$(submit "savio3_gpu" "a40_gpu3_normal" "gpu:A40:${g}" "$g" "$CPUS_A40")
    echo "savio3_gpu A40 (${g} GPU) -> $jid"
    JOBS+=("$jid")
done

# savio3_gpu GTX2080Ti
for g in 1; do
    jid=$(submit "savio3_gpu" "gtx2080_gpu3_normal" "gpu:GTX2080TI:${g}" "$g" "$CPUS_2080TI")
    echo "savio3_gpu GTX2080Ti (${g} GPU) -> $jid"
    JOBS+=("$jid")
done

# savio2_1080ti GTX1080Ti
for g in 1; do
    jid=$(submit "savio2_1080ti" "savio_normal" "gpu:GTX1080TI:${g}" "$g" "$CPUS_1080TI")
    echo "savio2_1080ti (${g} GPU) -> $jid"
    JOBS+=("$jid")
done

echo
echo "All jobs submitted. Logs will be written to ${LOG_DIR}/slurm-<jobid>.{out,err}"
echo "Check status with: squeue -j $(IFS=,; echo "${JOBS[*]}")"

exit 0