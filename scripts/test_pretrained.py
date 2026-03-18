import argparse
import glob
import json
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.baseline_model import compute_metrics


PRETRAINED_URL = "http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar"
PRETRAINED_MODELS = [
	"cityscapes_cut_pretrained",
	"cityscapes_fastcut_pretrained",
	"horse2zebra_cut_pretrained",
	"horse2zebra_fastcut_pretrained",
	"cat2dog_cut_pretrained",
	"cat2dog_fastcut_pretrained",
]


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
	cmd = [
		sys.executable,
		os.path.join(cut_dir, "test.py"),
		"--dataroot", dataroot,
		"--name", model_name,
		"--model", "cut",
		"--direction", direction,
		"--results_dir", results_dir,
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

	image_dir = os.path.join(results_dir, model_name, f"{phase}_latest", "images")
	fake_paths = glob.glob(os.path.join(image_dir, "*fake_A*")) + glob.glob(os.path.join(image_dir, "*fake_B*"))
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
	fake_a = sorted(glob.glob(os.path.join(output_dir, "*fake_A*")))
	fake_b = sorted(glob.glob(os.path.join(output_dir, "*fake_B*")))
	a_paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
	b_paths = sorted(glob.glob(os.path.join(tgt_dir, "*.jpg")))

	if direction == "AtoB":
		if not fake_b:
			raise RuntimeError("No fake_B outputs found for AtoB metric evaluation.")
		return compute_metrics(tgt_dir, output_dir, a_paths, fake_b, device)
	if not fake_a:
		raise RuntimeError("No fake_A outputs found for BtoA metric evaluation.")
	return compute_metrics(src_dir, output_dir, b_paths, fake_a, device)


def main():
	parser = argparse.ArgumentParser(description="Test all CUT pretrained models on movie-game style transfer data.")
	parser.add_argument("--project-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	parser.add_argument("--cut-dir", default=None)
	parser.add_argument("--dataroot", default=None, help="CUT dataroot containing testA/testB or trainA/trainB")
	parser.add_argument("--results-dir", default=None)
	parser.add_argument("--direction", choices=["AtoB", "BtoA", "both"], default="both")
	parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
	args = parser.parse_args()

	project_root = os.path.abspath(args.project_root)
	cut_dir = args.cut_dir or os.path.join(project_root, "external", "contrastive-unpaired-translation")
	dataroot = args.dataroot or os.path.join(project_root, "output", "cut_data")
	results_dir = args.results_dir or os.path.join(project_root, "output", "pretrained_eval")

	if not os.path.isdir(cut_dir):
		raise FileNotFoundError(f"CUT directory not found: {cut_dir}")
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
	print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
	main()

