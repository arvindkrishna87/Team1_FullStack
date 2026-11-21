"""Run main.py for one Objective 2 split, writing to its own fixed run folder.

main.py names run folders by timestamp, so array tasks that start in the same
second collide. This wrapper swaps in a per-split folder (obj2_<netid>/rXX_fY)
and then runs main.main() unchanged with the remaining arguments.
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--runs_root", default="/projects/e32998/MIL_training/final_runs/obj2_chj4322")
args, rest = ap.parse_known_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M

out = os.path.join(args.runs_root, args.tag)

def _fixed_run_dir(*_a, **_k):
    os.makedirs(out, exist_ok=True)
    print(f"Created run directory: {out}")
    return out

M.create_run_directory = _fixed_run_dir
sys.argv = ["main.py", "--load_splits", args.split] + rest
M.main()
