#!/bin/bash

echo "Folder Name,File Count,Subdirectory Count" > file_counts.csv

for dir in ../results_A100D-2-20C/*; do
  # Remove trailing slash from directory name
  dir=${dir%/}
  
  # Count files and directories in directory
  file_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
  subdirectory_count=$(find "$dir" -maxdepth 1 -type d | wc -l)

  # Output results to CSV file
  echo "$dir,$file_count,$((subdirectory_count-1))" >> file_counts.csv
done
