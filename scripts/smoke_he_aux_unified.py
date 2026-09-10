#!/usr/bin/env python3
"""Real-data smoke test for the HE-anchored unified model.

For each of the three example configs (`configs/he_aux_unified/*.json`) this
script, on **real C16 features**:

  1. resolves the feature directories from the config and asserts every
     configured stain is actually on disk (a listed-but-absent stain must fail
     loudly, never silently shrink the modality list);
  2. builds the model through the *exact* path `train.py` uses
     (`build_he_aux_unified_from_config`) and loads real slides through
     `C16MultimodalDataset`, asserting the dataset emits one feature tensor per
     configured stain with identical patch counts;
  3. runs a real forward + CE backward on GPU and reports, per auxiliary branch,
     the parameter count, grad-norm and whether the optimizer covers it;
  4. checks the missing-stain error path.

Paths resolve from this file's location (repo root); the feature root, label
files and device are all CLI arguments, so nothing here is machine-specific.

Usage:
  python scripts/smoke_he_aux_unified.py [--config NAME ...] [--device cuda:2]
      [--feature-base-dir DIR] [--train-labels CSV] [--val-labels CSV]
"""
import argparse, json, sys, traceback
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from data.c16_multimodal_dataset import C16MultimodalDataset
from models.he_aux_unified import build_he_aux_unified_from_config

DEFAULT_CONFIG_DIR = PROJECT / "configs" / "he_aux_unified"
SMOKE_MAX_PATCHES = 256        # keep the smoke run small; real runs use 2500


def load_cfg(name, config_dir=None):
    return json.loads((Path(config_dir or DEFAULT_CONFIG_DIR) / f"{name}.json")
                      .read_text())


def apply_overrides(data_cfg, args):
    """CLI overrides for the real-data locations; config values are the default."""
    if args.feature_base_dir:
        data_cfg['feature_base_dir'] = args.feature_base_dir
    if args.train_labels:
        data_cfg['train_label_file'] = args.train_labels
    if args.val_labels:
        data_cfg['val_label_file'] = args.val_labels
    return data_cfg


def resolve_feature_dirs(data_cfg, mapping_override=None):
    base = Path(data_cfg['feature_base_dir'])
    mapping = dict(data_cfg.get('dir_mapping') or {})
    if mapping_override:
        mapping.update(mapping_override)
    return {m: base / mapping.get(m, f"C16_{m}_features")
            for m in data_cfg['modalities']}


def check_stain_dirs(feature_dirs, strict=True):
    """Report on-disk slide counts per stain; raise if a stain is missing."""
    report, missing = {}, []
    for mod, d in feature_dirs.items():
        if not d.is_dir():
            missing.append((mod, str(d)))
            report[mod] = 0
            continue
        report[mod] = sum(1 for _ in d.rglob('*.pt'))
    if strict and missing:
        raise FileNotFoundError(
            "configured stain(s) absent on disk: "
            + ', '.join(f"{m} → {p}" for m, p in missing)
            + " (a configured stain must never be silently skipped)")
    return report


def run_case(name, device, args):
    cfg = load_cfg(name, args.config_dir)
    data_cfg = apply_overrides(cfg['data'], args)
    model_cfg = cfg['model']
    mods = data_cfg['modalities']
    lines = [f"── {name}: modalities={mods} device={device}"]

    # 1. on-disk presence of every configured stain
    feature_dirs = resolve_feature_dirs(data_cfg)
    counts = check_stain_dirs(feature_dirs)
    lines.append("   stain feature files: "
                 + ', '.join(f"{m}={counts[m]}" for m in mods))
    assert all(counts[m] > 0 for m in mods), "every configured stain must exist"

    # 2. real dataset — one feature tensor per configured stain
    strict = bool(data_cfg.get('strict_modalities', True))
    ds = C16MultimodalDataset(
        feature_dirs={m: str(feature_dirs[m]) for m in mods},
        label_file=data_cfg['train_label_file'],
        max_patches=args.max_patches,
        preload=False, verbose=False, sampling='random',
        sample_seed=data_cfg.get('sample_seed', 42), per_epoch=False,
        strict_modalities=strict)
    lines.append(f"   dataset: {len(ds)} train slides, "
                 f"{ds.num_modalities} modalities {ds.modalities}")
    assert ds.num_modalities == len(mods)
    assert ds.modalities == mods, f"dataset order {ds.modalities} != config {mods}"

    sample = ds[0]
    feats = [sample['features'][m] for m in mods]
    shapes = [tuple(f.shape) for f in feats]
    assert len(feats) == len(mods)
    assert len({f.shape[0] for f in feats}) == 1, \
        f"patch counts differ across stains: {shapes}"

    # 3. model + real forward/backward
    model = build_he_aux_unified_from_config(model_cfg, data_cfg).to(device)
    model.train()
    info = model.get_config()
    n_params = sum(p.numel() for p in model.parameters())
    lines.append(f"   model: {type(model).__name__} aux={info['aux_stains']} "
                 f"mil={info['mil_cfg']['name']} params={n_params:,}")
    lines.append(f"   encoder cfg source: {info['encoder_cfg_source']}")

    named = dict(model.named_parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt_ids = {id(p) for g in opt.param_groups for p in g['params']}
    assert opt_ids == {id(p) for p in model.parameters()}, "optimizer coverage"

    slides = [ds[i] for i in range(min(3, len(ds)))]
    opt.zero_grad()
    losses, per_branch = [], {s: [] for s in model.aux_stains}
    for s in slides:
        x = [s['features'][m].to(device) for m in mods]
        logits = model(x)[0]
        assert logits.shape == (1, data_cfg['num_classes']), logits.shape
        loss = F.cross_entropy(logits, torch.tensor([s['label']], device=device))
        loss.backward()
        losses.append(loss.item())
        for m in model.aux_stains:
            g = [named[k].grad for k in named
                 if k.startswith(f'cross_branches.{m}.') and named[k].grad is not None]
            per_branch[m].append(sum(float(x.norm()) for x in g))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for m in model.aux_stains:
        bp = [k for k in named if k.startswith(f'cross_branches.{m}.')]
        assert bp, f"branch {m} has no parameters"
        assert all(id(named[k]) in opt_ids for k in bp), f"{m} not in the optimizer"
        assert all(named[k].grad is not None for k in bp), f"{m} has a param w/o grad"
        assert all(torch.isfinite(named[k].grad).all() for k in bp), \
            f"{m} has a non-finite grad"
        n_tensors = len(bp)
        n_elems = sum(named[k].numel() for k in bp)
        lines.append(
            f"   branch[{m}]: {n_tensors} tensors / {n_elems:,} params, "
            f"grad_norm per slide = "
            f"[{', '.join(f'{v:.3e}' for v in per_branch[m])}], "
            f"prototype_initialized={bool(model.cross_branches[m].prototype_initialized)}")

    # HE branch + MIL also trained
    for pref in ('rrt.HE.', 'patch_to_emb.HE.', 'mil.'):
        assert any(k.startswith(pref) for k in named), f"missing {pref}"
    lines.append(f"   losses over {len(losses)} slides: "
                 + ', '.join(f"{v:.4f}" for v in losses))

    # 4. a configured-but-absent stain must raise, not shrink the dataset
    aux_mod = mods[-1]
    bad_dir = {m: str(feature_dirs[m]) for m in mods}
    bad_dir[aux_mod] = str(feature_dirs[aux_mod]) + "__MISSING"
    try:
        check_stain_dirs({k: Path(v) for k, v in bad_dir.items()})
        raise AssertionError("on-disk check must flag the absent stain")
    except FileNotFoundError:
        pass
    try:
        C16MultimodalDataset(feature_dirs=bad_dir,
                             label_file=data_cfg['train_label_file'],
                             max_patches=8, preload=False, verbose=False,
                             strict_modalities=True)
        raise AssertionError("strict dataset must refuse a missing stain dir")
    except FileNotFoundError as e:
        assert aux_mod in str(e) and 'C16_' in str(e), str(e)
    # ... and the lenient (historical) mode still silently yields no samples
    lenient = C16MultimodalDataset(feature_dirs=bad_dir,
                                   label_file=data_cfg['train_label_file'],
                                   max_patches=8, preload=False, verbose=False,
                                   strict_modalities=False)
    lines.append(f"   missing-stain check: strict → FileNotFoundError ✔, "
                 f"lenient → {len(lenient)} samples (historical behaviour)")

    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', nargs='*',
                    default=['he_only', 'he_pr', 'he_4aux'],
                    help='config basenames under --config-dir')
    ap.add_argument('--config-dir', default=str(DEFAULT_CONFIG_DIR),
                    help='directory holding the example configs')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-patches', type=int, default=SMOKE_MAX_PATCHES)
    ap.add_argument('--feature-base-dir', default=None,
                    help='override data.feature_base_dir')
    ap.add_argument('--train-labels', default=None,
                    help='override data.train_label_file')
    ap.add_argument('--val-labels', default=None,
                    help='override data.val_label_file')
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"repo root  : {PROJECT}")
    print(f"config dir : {args.config_dir}")
    print(f"device     : {dev}\n")
    torch.manual_seed(42)
    ok = 0
    for name in args.config:
        try:
            for line in run_case(name, dev, args):
                print(line)
            print(f"[PASS] smoke {name}\n")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] smoke {name} — {type(e).__name__}: {e}")
            traceback.print_exc()
            print()
    print(f"{ok}/{len(args.config)} configs passed the real-data smoke test")
    return 0 if ok == len(args.config) else 1


if __name__ == '__main__':
    raise SystemExit(main())
