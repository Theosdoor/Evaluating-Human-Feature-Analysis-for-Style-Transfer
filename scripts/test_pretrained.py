import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.baseline_model import compute_metrics
from src.data import flat_paths_by_domain, get_data_split


PRETRAINED_URL = "http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar"
PRETRAINED_MODELS = [
	"cityscapes_cut_pretrained",
	"cityscapes_fastcut_pretrained",
	"horse2zebra_cut_pretrained",
	"horse2zebra_fastcut_pretrained",
	"cat2dog_cut_pretrained",
	"cat2dog_fastcut_pretrained",
]


def _image_paths(dir_path):
	patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
	paths = []
	for pattern in patterns:
		paths.extend(glob.glob(os.path.join(dir_path, pattern)))
	return sorted(paths)


def _fake_paths_for_direction(output_dir, direction):
	"""
	CUT test.py writes outputs into subdirectories under images/:
	  images/fake_B/  (for AtoB)
	  images/fake_A/  (for BtoA)
	rather than flat files with 'fake_B' in the filename.
	Try both the subdirectory layout and the flat layout as a fallback.
	"""
	subdir_name = "fake_B" if direction == "AtoB" else "fake_A"
	subdir = os.path.join(output_dir, subdir_name)
	if os.path.isdir(subdir):
		paths = _image_paths(subdir)
		if paths:
			return paths, subdir

	# Fallback: flat files with domain in the name
	glob_pattern = f"*{subdir_name}*"
	paths = sorted(glob.glob(os.path.join(output_dir, glob_pattern)))
	return paths, output_dir


def _latest_classification_dir(project_root):
	base = os.path.join(project_root, "output", "classifications")
	if not os.path.isdir(base):
		raise FileNotFoundError(f"Classification base dir not found: {base}")
	dirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
	if not dirs:
		raise FileNotFoundError(f"No classification runs found in: {base}")
	return max(dirs, key=os.path.getmtime)


def _build_split_dataroot(project_root, cls_dir, split_ratio, seed, exclude_classes_csv, split_name):
	exclude_classes = [c.strip() for c in exclude_classes_csv.split(",") if c.strip()]
	split = get_data_split(
		cls_dir,
		train_split=split_ratio,
		seed=seed,
		exclude_classes=exclude_classes,
	)
	train_game, train_movie = flat_paths_by_domain(split["train"])

	if not train_game or not train_movie:
		raise RuntimeError(
			"Split produced empty game or movie train set. "
			"Try reducing exclusions or using a lower split ratio."
		)

	name = split_name or f"split_{str(split_ratio).replace('.', 'p')}_{os.path.basename(cls_dir)}"
	dataroot = os.path.join(project_root, "output", "pretrained_eval_dataroot", name)
	train_a = os.path.join(dataroot, "trainA")
	train_b = os.path.join(dataroot, "trainB")

	if os.path.isdir(dataroot):
		shutil.rmtree(dataroot)
	os.makedirs(train_a, exist_ok=True)
	os.makedirs(train_b, exist_ok=True)

	for i, src in enumerate(train_game):
		dst = os.path.join(train_a, f"game_{i:06d}{os.path.splitext(src)[1].lower()}")
		os.symlink(os.path.abspath(src), dst)
	for i, src in enumerate(train_movie):
		dst = os.path.join(train_b, f"movie_{i:06d}{os.path.splitext(src)[1].lower()}")
		os.symlink(os.path.abspath(src), dst)

	print(f"Built split dataroot from {cls_dir}")
	print(f"  trainA (game):  {len(train_game)}")
	print(f"  trainB (movie): {len(train_movie)}")
	print(f"  dataroot: {dataroot}")
	return dataroot


def ensure_pretrained_models(cut_dir):
	tar_path = os.path.join(cut_dir, "pretrained_models.tar")
	checkpoints_dir = os.path.join(cut_dir, "checkpoints")
	probe = os.path.join(checkpoints_dir, "horse2zebra_cut_pretrained", "latest_net_G.pth")

	if not os.path.exists(tar_path):
		print(f"Downloading pretrained archive -> {tar_path}")
		subprocess.run(["wget", "-O", tar_path, PRETRAINED_URL], check=True)

	if not os.path.exists(probe):
		print("Extracting pretrained archive...")
		subprocess.run(["tar", "-xf", tar_path], check=True, cwd=cut_dir)
	else:
		print("Pretrained checkpoints already available. Skipping extraction.")


def run_test(cut_dir, model_name, dataroot, results_dir, direction, gpu_ids):
	phase = "test" if os.path.isdir(os.path.join(dataroot, "testA")) else "train"
	results_dir_abs = os.path.abspath(results_dir)
	cmd = [
		sys.executable,
		os.path.join(cut_dir, "test.py"),
		"--dataroot", dataroot,
		"--name", model_name,
		"--model", "cut",
		"--direction", direction,
		"--results_dir", results_dir_abs,
		"--checkpoints_dir", os.path.join(cut_dir, "checkpoints"),
		"--gpu_ids", gpu_ids,
		"--load_size", "256",
		"--crop_size", "256",
		"--no_flip",
		"--num_test", "9999",
		"--phase", phase,
		"--eval",
	]
	print(f"\nRunning: model={model_name} direction={direction}")
	subprocess.run(cmd, check=True, cwd=cut_dir)

	image_dir = os.path.join(results_dir_abs, model_name, f"{phase}_latest", "images")
	fake_paths, _ = _fake_paths_for_direction(image_dir, direction)
	return {
		"model": model_name,
		"direction": direction,
		"phase": phase,
		"output_dir": image_dir,
		"num_fake_images": len(fake_paths),
	}


def evaluate_metrics(dataroot, output_dir, direction, device, phase):
	src_dir = os.path.join(dataroot, f"{phase}A")
	tgt_dir = os.path.join(dataroot, f"{phase}B")
	a_paths = _image_paths(src_dir)
	b_paths = _image_paths(tgt_dir)

	fake_paths, fake_dir = _fake_paths_for_direction(output_dir, direction)

	if not fake_paths:
		domain = "fake_B" if direction == "AtoB" else "fake_A"
		raise RuntimeError(f"No {domain} outputs found in {output_dir} for {direction} metric evaluation.")

	if direction == "AtoB":
		return compute_metrics(tgt_dir, fake_dir, a_paths, fake_paths, device)
	return compute_metrics(src_dir, fake_dir, b_paths, fake_paths, device)


def save_fake_video(output_dir, direction, fps=24):
	fake_paths, _ = _fake_paths_for_direction(output_dir, direction)

	if not fake_paths:
		domain = "fake_B" if direction == "AtoB" else "fake_A"
		raise RuntimeError(f"No translated {domain} frames found in {output_dir} for video export.")

	first = cv2.imread(fake_paths[0])
	if first is None:
		raise RuntimeError(f"Could not read first fake image: {fake_paths[0]}")

	h, w = first.shape[:2]
	video_path = os.path.join(output_dir, f"translated_{direction}.mp4")
	writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
	if not writer.isOpened():
		raise RuntimeError(f"Could not open video writer for: {video_path}")

	for path in fake_paths:
		frame = cv2.imread(path)
		if frame is None:
			continue
		if frame.shape[:2] != (h, w):
			frame = cv2.resize(frame, (w, h))
		writer.write(frame)
	writer.release()
	print(f"Saved video -> {video_path}")
	return video_path


def main():
	parser = argparse.ArgumentParser(description="Test all CUT pretrained models on movie-game style transfer data.")
	parser.add_argument("--project-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	parser.add_argument("--cut-dir", default=None)
	parser.add_argument("--dataroot", default=None, help="CUT dataroot containing testA/testB or trainA/trainB")
	parser.add_argument("--results-dir", default=None)
	parser.add_argument("--direction", choices=["AtoB", "BtoA", "both"], default="both")
	parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
	parser.add_argument("--use-data-split", action="store_true",
	                    help="Build dataroot from src.data.get_data_split before running pretrained tests")
	parser.add_argument("--cls-dir", default=None,
	                    help="Classification run directory (default: latest under output/classifications)")
	parser.add_argument("--split-train-ratio", type=float, default=1.0,
	                    help="train_split passed to get_data_split when --use-data-split is set")
	parser.add_argument("--split-seed", type=int, default=42)
	parser.add_argument("--exclude-classes", default="others",
	                    help="Comma-separated classes to exclude in split mode")
	parser.add_argument("--split-name", default=None,
	                    help="Optional name for generated split dataroot directory")
	parser.add_argument("--save-video", action="store_true",
	                    help="If set, save translated fake frames as MP4 per model+direction")
	parser.add_argument("--video-fps", type=float, default=24.0,
	                    help="FPS for saved MP4 videos (used with --save-video)")
	args = parser.parse_args()

	project_root = os.path.abspath(args.project_root)
	cut_dir = args.cut_dir or os.path.join(project_root, "external", "contrastive-unpaired-translation")
	dataroot = args.dataroot or os.path.join(project_root, "output", "cut_data")
	results_dir = args.results_dir or os.path.join(project_root, "output", "pretrained_eval")

	if not os.path.isdir(cut_dir):
		raise FileNotFoundError(f"CUT directory not found: {cut_dir}")

	if args.use_data_split:
		cls_dir = os.path.abspath(args.cls_dir) if args.cls_dir else _latest_classification_dir(project_root)
		dataroot = _build_split_dataroot(
			project_root,
			cls_dir,
			args.split_train_ratio,
			args.split_seed,
			args.exclude_classes,
			args.split_name,
		)
		if args.results_dir is None:
			split_tag = str(args.split_train_ratio).replace('.', 'p')
			results_dir = os.path.join(project_root, "output", f"pretrained_eval_split_{split_tag}")
		print(f"Using data-split dataroot: {dataroot}")
	else:
		print(f"Using dataroot: {dataroot}")

	results_dir = os.path.abspath(results_dir)

	if not os.path.isdir(dataroot):
		raise FileNotFoundError(f"Dataroot not found: {dataroot}")
	if not (
		os.path.isdir(os.path.join(dataroot, "testA")) and os.path.isdir(os.path.join(dataroot, "testB"))
	) and not (
		os.path.isdir(os.path.join(dataroot, "trainA")) and os.path.isdir(os.path.join(dataroot, "trainB"))
	):
		raise FileNotFoundError(f"Dataroot must contain testA/testB or trainA/trainB: {dataroot}")

	os.makedirs(results_dir, exist_ok=True)
	gpu_ids = "0" if args.device == "cuda" else "-1"

	ensure_pretrained_models(cut_dir)

	directions = ["AtoB", "BtoA"] if args.direction == "both" else [args.direction]
	summary = []
	for model_name in PRETRAINED_MODELS:
		for direction in directions:
			try:
				run_info = run_test(cut_dir, model_name, dataroot, results_dir, direction, gpu_ids)
				if args.save_video:
					try:
						run_info["video_path"] = save_fake_video(
							run_info["output_dir"], direction, fps=args.video_fps
						)
					except Exception as video_exc:
						run_info["video_error"] = str(video_exc)
				try:
					metrics = evaluate_metrics(
						dataroot,
						run_info["output_dir"],
						direction,
						args.device,
						run_info["phase"],
					)
					run_info["metrics"] = metrics
				except Exception as metric_exc:
					run_info["metrics_error"] = str(metric_exc)
				run_info["status"] = "ok"
			except subprocess.CalledProcessError as exc:
				run_info = {
					"model": model_name,
					"direction": direction,
					"status": "failed",
					"return_code": exc.returncode,
				}
			summary.append(run_info)

	summary_path = os.path.join(results_dir, "pretrained_summary.json")
	with open(summary_path, "w") as f:
		json.dump(summary, f, indent=2)

	print("\nDone. Summary:")
	for item in summary:
		print(item)
	print(f"Results directory -> {results_dir}")
	print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
	main()