#!/bin/bash
#SBATCH --job-name=ACV
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=24G

cd /home2/nchw73/Year4/ACV_cswk
uv sync
source .venv/bin/activate

echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# Run the experiments
# python scripts/train_gcn.py \
#     --extract-dir output/extracted_humans/20260324-185427 \
#     --label-source manual \
#     --annotations output/manual_annotated/20260324-185427-1
# python scripts/train_gcn.py \
#         --extract-dir  output/extracted_humans/20260324-185427 \
#         --label-source rule \
#         --pose-model   models/yolo26m-pose.pt 

# python scripts/train_gcn.py --ablation \
#     --extract-dir  output/extracted_humans/20260324-185427 \
#     --init-cls-dir output/init_classifications/20260324-185427-1 \
#     --pose-model   models/yolo26m-pose.pt
python3 nb_main.py
# python3 scripts/nb_figures.py