# ComputerVisionBenchmark

This repository benchmarks Detectron2 object-detection models on a Rumex weed dataset under different GPU profiles and batch-size configurations. It includes utilities for launching training runs, monitoring GPU/CPU utilization, and analyzing the generated results directories.

## Repository structure
- `train.py`: Detectron2 training and evaluation script that registers the dataset, trains a selected model, logs metrics, saves artifacts, and exports sample inference images.
- `run_training.sh`: Helper script to iterate through multiple architectures, batch sizes, and iteration counts when running benchmarks.
- `gpu_cpu_monitoring.sh`: Collects GPU and CPU resource metrics with `nvidia-smi` and `top`, writing results to CSV/text files per profile.
- `run_benchmark_profile.sh`: Intended entry point for running a benchmark profile; adjust the script to orchestrate monitoring and training runs for your environment.
- `analysis/`: Post-processing utilities for summarizing results directories.
  - `count_files.sh`: Generates `analysis/file_counts.csv` with file/subdirectory counts for each results folder.
  - `analyze_possibilties.R`: Produces `analysis/Model_Vs_BatchSize_Possibility.csv` indicating which model/batch-size combinations produced sufficient output.
- `Benchmark_notebook.ipynb`: Notebook for exploratory benchmarking and visualization.
- `BenchMarkDocumentation.md`: Notes on initial benchmarking ideas and model choices.

## Requirements
- Python environment with Detectron2 and dependencies such as PyTorch, torchvision, OpenCV, matplotlib, pandas, numpy, and mlflow.
- CUDA-enabled GPU for training (training script assumes `torch.cuda.is_available()`).
- Dataset mounted under `./images/<dataset_name>/` with COCO-format annotations.
- Optional: R (for the analysis script) and standard UNIX utilities (`find`, `wc`, `top`, `nvidia-smi`).

## Dataset layout
The training script expects the following directory structure relative to the repository root:

```
images/
  20220823_HaldenSued_S_10_F_50_O_stra_ID1/
    1_images/
      Chunks/              # Image tiles used for training/validation/testing
    3_annotations/
      train_chuncks.json
      val_chuncks.json
      test_chuncks.json
```

Update `train.py` if your dataset path or partition names differ.

## Running a single training job
Invoke `train.py` with the model architecture (relative to Detectron2's `COCO-Detection` configs), images-per-batch, and iteration count:

```bash
python train.py --arch faster_rcnn_R_50_FPN_1x.yaml --ipb 8 --iter 500
```

Artifacts are saved to `results_A100D-2-20C/<model>_chunks_IPB_<BATCH>_MI_<ITER>/`, including model checkpoints, AP metrics, timing metadata, sample inference images, and loss/accuracy plots.

## Running benchmark sweeps
`run_training.sh` contains arrays of architectures (`archs`), batch sizes (`IMS_PER_BATCH`), and iteration counts (`MAX_ITER`). Customize these arrays, then execute:

```bash
bash run_training.sh
```

Each combination triggers a call to `train.py` with the specified parameters. Review and adjust the script to match your available GPU memory and desired model set.

## Monitoring GPU/CPU utilization
To capture resource usage while training, run the monitoring script in a separate shell:

```bash
bash gpu_cpu_monitoring.sh
```

Metrics are appended to `results_A100D-7-80C/gpu_resources.csv` and `results_A100D-7-80C/cpu_resources.txt` by default; change `PROFILE_NAME` inside the script to separate runs by hardware profile.

## Analyzing results
After training, generate summaries of the results directory:

```bash
cd analysis
bash count_files.sh
Rscript analyze_possibilties.R
```

`file_counts.csv` reports how many files each experiment directory contains, and `Model_Vs_BatchSize_Possibility.csv` flags which model/batch-size combinations produced complete outputs.

## Notes
- The provided scripts assume a UNIX-like environment with NVIDIA GPUs. Adjust paths and commands to fit your setup.
- Detectron2 configuration defaults (dataset names, augmentations, learning rate, ROI head settings, etc.) are defined in `train.py` and can be modified to explore additional scenarios.
- Images are available under request.
