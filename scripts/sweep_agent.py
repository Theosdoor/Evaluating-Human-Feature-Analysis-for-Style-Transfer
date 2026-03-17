"""
scripts/sweep_agent.py
Single-trial agent for the wandb Bayesian sweep (see sweep.yaml).

Usage: called automatically by `wandb agent <sweep-id>`
"""

import os
import shutil
import subprocess
import sys

import wandb
from cleanfid import fid as cleanfid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.baseline_model import make_inference_dataroot, run_inference

CUT_DIR  = os.path.join(PROJECT_ROOT, "contrastive-unpaired-translation")
DATA_DIR = os.path.join(PROJECT_ROOT, "output", "cut_data")
DEVICE   = "cuda"

with wandb.init() as run:
    cfg = run.config
    exp_name = f"sweep_{run.id}"
    ckpt_dir = os.path.join(CUT_DIR, "checkpoints", exp_name)

    cmd = [
        sys.executable, os.path.join(CUT_DIR, "train.py"),
        "--dataroot",        DATA_DIR,
        "--name",            exp_name,
        "--model",           "cut",
        "--CUT_mode",        "CUT",
        "--direction",       "AtoB",
        "--n_epochs",        "50",
        "--n_epochs_decay",  "50",
        "--batch_size",      "4",
        "--load_size",       "286",
        "--crop_size",       "256",
        "--gpu_ids",         "0",
        "--checkpoints_dir", os.path.join(CUT_DIR, "checkpoints"),
        "--no_html",
        "--display_id",      "0",
        "--nce_idt",         "True",
        "--save_epoch_freq", "999",
        "--lambda_NCE",      str(cfg.lambda_NCE),
        "--nce_T",           str(cfg.nce_T),
        "--lr",              str(cfg.lr),
    ]
    subprocess.run(cmd, check=True, cwd=CUT_DIR)

    testA = os.path.join(DATA_DIR, "testA")
    testB = os.path.join(DATA_DIR, "testB")
    results_dir = os.path.join(PROJECT_ROOT, "output", "sweep_results", run.id)

    run_inference(
        CUT_DIR, exp_name,
        make_inference_dataroot(testA, testB),
        results_dir, "AtoB", DEVICE,
    )

    fid_score = cleanfid.compute_fid(
        testB, os.path.join(results_dir, "fake"), device=DEVICE, verbose=False
    )
    wandb.log({"fid_g2m": fid_score})
    print(f"FID: {fid_score:.2f}")

    shutil.rmtree(ckpt_dir, ignore_errors=True)
