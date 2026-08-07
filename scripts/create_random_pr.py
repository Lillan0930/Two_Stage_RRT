#!/usr/bin/env python3
"""
Create a randomly-shuffled PR feature directory using symlinks.

Each .pt file in the shuffled directory is a symlink to a *different* patient's
real PR feature.  The shuffle uses a fixed seed (9999) so all Random-PR
experiments use the same permutation — the comparison is: does the specific
HE↔PR pairing matter, or is any PR signal equally useful?

Output:  /home/Public/lillan/features_result/C16_features/C16_PR_random_features/
"""

import os, random, shutil
from pathlib import Path

SRC = Path("/home/Public/lillan/features_result/C16_features/C16_PR_features")
DST = Path("/home/Public/lillan/features_result/C16_features/C16_PR_random_features")
SHUFFLE_SEED = 9999

# Collect all (subdir, filename) pairs
files = []
for subdir in ["normal", "tumor", "test"]:
    d = SRC / subdir
    if d.is_dir():
        for f in sorted(d.glob("*.pt")):
            files.append((subdir, f.name))

print(f"Found {len(files)} PR feature files")

# Shuffle with fixed seed
rng = random.Random(SHUFFLE_SEED)
shuffled = list(files)
rng.shuffle(shuffled)

# Verify it's actually shuffled
same = sum(1 for a, b in zip(files, shuffled) if a == b)
print(f"Files keeping original position after shuffle: {same}/{len(files)}")

# Create shuffled symlink directory
if DST.exists():
    print(f"Removing existing {DST} ...")
    shutil.rmtree(DST)

for (src_subdir, src_name), (dst_subdir, dst_name) in zip(files, shuffled):
    dst_dir = DST / dst_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)

    # The shuffled dir uses dst_name, but links to src's actual content
    src_path = SRC / src_subdir / src_name
    dst_path = dst_dir / dst_name

    # Relative symlink so the directory is portable
    rel_src = os.path.relpath(src_path, dst_path.parent)
    os.symlink(rel_src, dst_path)

print(f"Created {DST} with {len(files)} symlinked files")

# Quick verify: load one file and check it works
import torch
test_file = list(DST.rglob("*.pt"))[0]
t = torch.load(str(test_file), map_location="cpu", weights_only=True)
print(f"Verified: {test_file.name} → shape {t.shape}, dtype {t.dtype}")
print("Done!")
