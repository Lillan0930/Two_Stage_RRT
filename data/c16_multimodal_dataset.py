"""
C16 Multimodal Feature Dataset

支持 C16 数据结构：{feature_dir}/{normal,tumor,test}/file.pt
每个模态下有 normal/ tumor/ test/ 三个子目录，含 .pt 特征文件。

Sampling modes:
  - 'first':  取前 max_patches 个 patch（确定性，有空间偏差）
  - 'random': 稳定 hash 随机采样
    - per_epoch=False → 固定 deterministic（Val / Test 用）
    - per_epoch=True  → seed = base_seed + stable_hash(slide_id) + epoch（Train 用）

多模态时 indices 在 modality loop 外生成一次，保证 HE/PR 使用同一组 patch。
Hash 使用 hashlib.md5，不受 Python 进程 hash randomization 影响。
"""

import os
import hashlib
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


def stable_slide_seed(slide_id: str, base_seed: int, epoch: int = 0) -> int:
    """Deterministic, cross-process stable seed for a slide.

    Uses MD5 so the same (slide_id, base_seed, epoch) always produces
    the same seed regardless of PYTHONHASHSEED or process boundary.
    """
    key = f"{slide_id}_{base_seed}_{epoch}"
    h = hashlib.md5(key.encode()).hexdigest()
    slide_seed = int(h[:8], 16)
    return slide_seed % (2**31 - 1)


class C16MultimodalDataset:
    """
    C16 多模态特征数据集。

    特征目录结构:
        C16_HE_features/
            normal/normal_001.pt ... normal_160.pt
            tumor/tumor_001.pt ... tumor_111.pt
            test/test_001.pt ... test_129.pt

    Parameters
    ----------
    sampling : 'first' | 'random'
    sample_seed : int
        Base seed for random sampling. Must be an int.
    per_epoch : bool
        If True, epoch is mixed into the seed (via set_epoch()).
        Train=True, Val/Test=False.
    """

    def __init__(self,
                 feature_dirs: Dict[str, str],
                 label_file: str,
                 max_patches: Optional[int] = None,
                 preload: bool = False,
                 verbose: bool = True,
                 sampling: str = 'first',
                 sample_seed: int = 0,
                 per_epoch: bool = False):
        self.feature_dirs = {k: Path(v) for k, v in feature_dirs.items()}
        self.modalities = list(feature_dirs.keys())
        self.num_modalities = len(self.modalities)
        self.max_patches = max_patches
        self.preload = preload
        self.verbose = verbose
        self.sampling = sampling
        self.sample_seed = sample_seed
        self.per_epoch = per_epoch
        self._epoch = 0

        if verbose:
            print(f"Initializing C16MultimodalDataset with {self.num_modalities} modalities: {self.modalities}")
            if sampling == 'random':
                tag = "per-epoch" if per_epoch else "fixed"
                print(f"  Sampling: random ({tag}, base_seed={sample_seed})")

        # 加载标签
        self.labels = {}
        self._load_labels(label_file)

        # 构建样本列表
        self.samples = []
        self._build_samples()

        # 预加载
        if self.preload:
            self.features_cache = {}
            self._preload_features()

    def set_epoch(self, epoch: int):
        """Set current epoch for per-epoch random sampling."""
        self._epoch = epoch

    def _build_indices(self, slide_id: str, total_patches: int) -> Optional[np.ndarray]:
        """Generate patch indices for one slide (called once, shared across modalities)."""
        if not self.max_patches or total_patches <= self.max_patches:
            return None  # no truncation needed

        if self.sampling == 'first':
            return np.arange(self.max_patches)

        # random sampling
        epoch = self._epoch if self.per_epoch else 0
        seed = stable_slide_seed(slide_id, self.sample_seed, epoch)
        rng = np.random.RandomState(seed)
        indices = rng.choice(total_patches, self.max_patches, replace=False)
        indices.sort()
        return indices

    def _load_labels(self, label_file):
        """加载 label CSV (slide_id,label)"""
        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            self.labels[str(row['slide_id'])] = int(row['label'])
        if self.verbose:
            print(f"Loaded {len(self.labels)} labels")

    def _build_samples(self):
        """构建样本列表，匹配所有模态的特征文件"""
        first_mod = self.modalities[0]
        first_dir = self.feature_dirs[first_mod]

        slide_feature_map = {}
        for subdir in ['normal', 'tumor', 'test']:
            subpath = first_dir / subdir
            if not subpath.exists():
                continue
            for f in subpath.glob('*.pt'):
                slide_id = f.stem
                if slide_id in self.labels:
                    slide_feature_map[slide_id] = f

        for slide_id in list(slide_feature_map.keys()):
            for mod in self.modalities[1:]:
                mod_dir = self.feature_dirs[mod]
                found = False
                for subdir in ['normal', 'tumor', 'test']:
                    candidate = mod_dir / subdir / f"{slide_id}.pt"
                    if candidate.exists():
                        found = True
                        break
                if not found:
                    del slide_feature_map[slide_id]
                    if self.verbose:
                        print(f"Warning: {slide_id} missing in modality {mod}, skipped")
                    break

        for slide_id in sorted(slide_feature_map.keys()):
            self.samples.append({
                'slide_id': slide_id,
                'label': self.labels[slide_id],
            })

        label_counts = {}
        for s in self.samples:
            lbl = s['label']
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if self.verbose:
            print(f"C16Dataset initialized: {len(self.samples)} samples")
            print(f"  Label distribution: {label_counts}")

    def _preload_features(self):
        """预加载所有特征到内存"""
        for mod in self.modalities:
            self.features_cache[mod] = {}
        for s in self.samples:
            sid = s['slide_id']
            for mod in self.modalities:
                mod_dir = self.feature_dirs[mod]
                for subdir in ['normal', 'tumor', 'test']:
                    candidate = mod_dir / subdir / f"{sid}.pt"
                    if candidate.exists():
                        t = torch.load(str(candidate), map_location='cpu', weights_only=True)
                        if t.dim() == 1:
                            t = t.unsqueeze(0)
                        self.features_cache[mod][sid] = t
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        slide_id = sample['slide_id']
        label = sample['label']

        # ── 1. Load first modality to determine total_patches ──
        first_mod = self.modalities[0]
        if self.preload and first_mod in self.features_cache:
            feat_first = self.features_cache[first_mod][slide_id]
        else:
            feat_first = self._load_feature(first_mod, slide_id)

        total_patches = feat_first.shape[0]

        # ── 2. Generate indices ONCE (shared across all modalities) ──
        patch_indices = self._build_indices(slide_id, total_patches)

        # ── 3. Load & truncate all modalities with same indices ──
        features = {}
        for i, mod in enumerate(self.modalities):
            if i == 0:
                feat = feat_first
            elif self.preload and mod in self.features_cache:
                feat = self.features_cache[mod][slide_id]
            else:
                feat = self._load_feature(mod, slide_id)

            # 辅助模态一致性检查（应用 patch_indices 之前）：patch 数量必须与
            # 第一模态一致，特征维度必须相同。共享 patch_indices 要求 HE/PR 的
            # patch 数量与顺序完全相同。
            if i > 0:
                if feat.shape[0] != total_patches:
                    raise ValueError(
                        f"{slide_id}: patch count mismatch between modalities: "
                        f"{first_mod}={total_patches}, {mod}={feat.shape[0]}. "
                        "Shared patch indices require identical patch counts and ordering."
                    )
                if feat.shape[-1] != feat_first.shape[-1]:
                    raise ValueError(
                        f"{slide_id}: feature dimension mismatch: "
                        f"{first_mod}={feat_first.shape[-1]}, "
                        f"{mod}={feat.shape[-1]}"
                    )

            if patch_indices is not None:
                feat = feat[patch_indices]
            features[mod] = feat

        return {
            'features': features,
            'label': label,
            'slide_id': slide_id,
        }

    def _load_feature(self, mod: str, slide_id: str) -> torch.Tensor:
        """Load a single .pt feature file."""
        mod_dir = self.feature_dirs[mod]
        for subdir in ['normal', 'tumor', 'test']:
            candidate = mod_dir / subdir / f"{slide_id}.pt"
            if candidate.exists():
                feat = torch.load(str(candidate), map_location='cpu', weights_only=True)
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                return feat
        raise FileNotFoundError(f"Feature not found for {slide_id} in {mod}")


def c16_multimodal_collate_fn(batch):
    """C16 collate: features by modality → list of tensors per sample"""
    if len(batch) == 0:
        return {'features': [], 'labels': torch.tensor([]), 'slide_ids': []}

    modalities = list(batch[0]['features'].keys())

    features_by_modality = []
    for mod in modalities:
        mod_features = [item['features'][mod] for item in batch]
        features_by_modality.append(mod_features)

    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    slide_ids = [item['slide_id'] for item in batch]

    return {
        'features': features_by_modality,
        'labels': labels,
        'slide_ids': slide_ids,
    }
