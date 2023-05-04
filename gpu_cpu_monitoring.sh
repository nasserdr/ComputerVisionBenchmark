#!/bin/bash

# Set the profile name
PROFILE_NAME="results_A100D-7-80C"

# Set the monitoring interval in seconds
MONITORING_INTERVAL=1

# Set the output file names
GPU_OUTPUT_FILE="${PROFILE_NAME}/gpu_resources.csv"
CPU_OUTPUT_FILE="${PROFILE_NAME}/cpu_resources.txt"

# Check if the output directory exists; create it if it does not
if [ ! -d "$PROFILE_NAME" ]; then
    mkdir -p $PROFILE_NAME
fi


MONITORING_INTERVAL=1
rm -f $GPU_OUTPUT_FILE
rm -f $CPU_OUTPUT_FILE

echo "Timestamp,GPU_Index,Name,Temperature_GPU,Utilization_GPU,Utilization_Memory,Memory_Total,Memory_Free,Memory_Used,Power_Draw" >> $GPU_OUTPUT_FILE

while true
do
    TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)

    nvidia-smi \
        --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw \
        --format=csv,noheader,nounits \
        >> $GPU_OUTPUT_FILE

    echo "Timestamp: $TIMESTAMP" >> $CPU_OUTPUT_FILE
    echo "========= CPU Monitoring =========" >> $CPU_OUTPUT_FILE
    top -b -n 1 | head -n 12 >> $CPU_OUTPUT_FILE
    echo "" >> $CPU_OUTPUT_FILE

    sleep $MONITORING_INTERVAL
done
