import os


def get_next_reclassify_dir(base_dir, run_name):
	"""
	Return a new classification directory path for reclassification runs.

	Example:
	If run_name is "20260314-195748" and existing directories include
	"20260314-195748-1" and "20260314-195748-3", this returns
	"20260314-195748-4".
	"""
	prefix = f"{run_name}-"
	max_idx = 0

	if os.path.isdir(base_dir):
		for name in os.listdir(base_dir):
			if not name.startswith(prefix):
				continue
			suffix = name[len(prefix):]
			if suffix.isdigit():
				max_idx = max(max_idx, int(suffix))

	return os.path.join(base_dir, f"{run_name}-{max_idx + 1}")