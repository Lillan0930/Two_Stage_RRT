"""
C17 Multimodal Feature Dataset

C17 数据结构（与 C16 的 `{normal,tumor,test}/slide.pt` 不同）::

    {feature_dir}/{patient_id}/{patient_id}_node_{k}.pt

例如::

    C17_HE_new_features/patient_000/patient_000_node_0.pt
    C17_PR_new_features/patient_000/patient_000_node_0.pt

一个 ``(patient_id, node_k)`` 就是一条样本；五个染色是同一个组织块的
不同染色切片，因此同名 ``.pt`` 之间**存在空间对应**。

Sampling modes（与 C16 数据集逐字段同源）:
  - 'first':  取前 max_patches 个 patch（确定性）
  - 'random': 稳定 hash 随机采样
    - per_epoch=False → 固定 deterministic（Val / Test 用）
    - per_epoch=True  → seed = base_seed + stable_hash(slide_id) + epoch（Train 用）

多模态时 indices 在 modality loop **之外**生成一次，五个染色共用同一组
patch 索引 —— 这是保持跨染色空间对应的前提。

严格性
------
``strict_modalities=True``（本流程固定开启）时：
  * 配置的染色目录缺失或为空 → 抛 ``FileNotFoundError``
  * 任何**有标签的 slide** 在任一染色下缺文件 → 抛 ``FileNotFoundError``
  * 同一 slide 跨染色 patch 数量不一致 → 抛 ``ValueError``
绝不静默取交集、静默删样本或静默缩小模态列表。
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def stable_slide_seed(slide_id: str, base_seed: int, epoch: int = 0) -> int:
    """Deterministic, cross-process stable seed for a slide (MD5, PYTHONHASHSEED-free)."""
    key = f"{slide_id}_{base_seed}_{epoch}"
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class C17MultimodalDataset:
    """C17 多模态特征数据集（``{patient}/{patient}_node_{k}.pt`` 布局）。"""

    def __init__(self,
                 feature_dirs: Dict[str, str],
                 label_file: str,
                 slide_ids: Optional[List[str]] = None,
                 max_patches: Optional[int] = None,
                 preload: bool = False,
                 verbose: bool = True,
                 sampling: str = 'first',
                 sample_seed: int = 0,
                 per_epoch: bool = False,
                 strict_modalities: bool = False,
                 require_same_patch_count: bool = True):
        """
        Args:
            slide_ids: 显式样本清单（本流程用 train/test 患者列表推导）。
                       ``None`` 表示用标签文件里的全部 slide。
            require_same_patch_count: 跨染色 patch 数必须一致，否则抛错
                                      （共享 patch 索引的前提）。
        """
        self.feature_dirs = {k: Path(v) for k, v in feature_dirs.items()}
        self.modalities = list(feature_dirs.keys())
        self.num_modalities = len(self.modalities)
        self.max_patches = max_patches
        self.preload = preload
        self.verbose = verbose
        self.sampling = sampling
        self.sample_seed = sample_seed
        self.per_epoch = per_epoch
        self.strict_modalities = strict_modalities
        self.require_same_patch_count = require_same_patch_count
        self.label_file = str(label_file)
        self._epoch = 0

        if verbose:
            print(f"Initializing C17MultimodalDataset with {self.num_modalities} "
                  f"modalities: {self.modalities}")
            if sampling == 'random':
                tag = "per-epoch" if per_epoch else "fixed"
                print(f"  Sampling: random ({tag}, base_seed={sample_seed})")

        self.labels = {}
        self._load_labels(label_file)

        self.samples = []
        self._build_samples(slide_ids)

        if self.preload:
            self.features_cache = {}
            self._preload_features()

    # ── setup ─────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int):
        """Set current epoch for per-epoch random sampling."""
        self._epoch = epoch

    def _load_labels(self, label_file):
        """加载 label CSV (slide_id,label)。"""
        df = pd.read_csv(label_file)
        if 'slide_id' not in df.columns:
            raise ValueError(f"{label_file}: 缺少 'slide_id' 列，实际列 = {list(df.columns)}")
        for _, row in df.iterrows():
            self.labels[str(row['slide_id'])] = int(row['label'])
        if self.verbose:
            print(f"Loaded {len(self.labels)} slide-level labels")

    def _feature_path(self, mod: str, slide_id: str) -> Path:
        """``{dir}/{patient_id}/{slide_id}.pt``；patient_id = slide_id 的前两段。"""
        patient_id = '_'.join(slide_id.split('_')[:2])
        return self.feature_dirs[mod] / patient_id / f"{slide_id}.pt"

    def _check_modality_dirs(self):
        """strict：每个配置的染色目录必须存在且含 .pt。"""
        empty = []
        for mod in self.modalities:
            d = self.feature_dirs[mod]
            n = sum(1 for _ in d.rglob('*.pt')) if d.is_dir() else 0
            if n == 0:
                empty.append((mod, str(d)))
        if empty:
            raise FileNotFoundError(
                "配置的染色目录缺失或为空: "
                + ', '.join(f"{m} → {p}" for m, p in empty)
                + f"; 拒绝静默缩小模态列表 {self.modalities}")

    def _build_samples(self, slide_ids):
        """构建样本列表。

        三类情况严格区分，**绝不静默**：

        1. 配置的染色目录缺失/为空 → ``FileNotFoundError``
        2. **模态不对称** —— 某个 slide 在染色 A 有特征、在染色 B 没有
           → ``FileNotFoundError``（这才是"缺模态"，会让共享 patch 索引失效）
        3. **标签表孤儿** —— 某个 slide 在所有配置染色下都**没有**特征
           → 显式排除，并把被排除的 ID 记进 ``self.excluded`` 打印出来。
           这不是模态问题，而是标签表比特征多了一行；历史 baseline 同样
           训练不到它（磁盘上没有文件），这里只是把这件事**明说**出来。

        情况 3 绝不静默：排除数量、ID、以及"标签表 N → 实际使用 M"都打印且
        可由调用方读取 ``self.excluded``。
        """
        if self.strict_modalities:
            self._check_modality_dirs()

        if slide_ids is None:
            wanted = sorted(self.labels)
        else:
            wanted = sorted(str(s) for s in slide_ids)

        # 每个染色实际可用的 slide 集合
        avail = {
            mod: {s for s in wanted if self._feature_path(mod, s).is_file()}
            for mod in self.modalities
        }

        # ── 情况 2：模态不对称（相对第一个染色） ──
        ref_mod, ref = self.modalities[0], avail[self.modalities[0]]
        asym = {}
        for mod in self.modalities[1:]:
            only_ref = sorted(ref - avail[mod])      # ref 有、该染色没有
            only_mod = sorted(avail[mod] - ref)      # 该染色有、ref 没有
            if only_ref or only_mod:
                asym[mod] = (only_ref, only_mod)
        if asym:
            detail = '; '.join(
                f"HE 有而 {m} 没有: {len(a)} (e.g. {', '.join(a[:3])}), "
                f"{m} 有而 HE 没有: {len(b)} (e.g. {', '.join(b[:3])})"
                for m, (a, b) in sorted(asym.items()))
            raise FileNotFoundError(
                f"{self.label_file!r}: 配置的染色之间覆盖不一致 — {detail}. "
                f"共享 patch 索引要求所有染色覆盖完全相同的 slide 集合；"
                f"拒绝静默取交集。")

        # ── 情况 3：标签表孤儿（所有染色都没有） ──
        orphans = sorted(set(wanted) - ref)
        self.excluded = orphans
        if orphans:
            msg = (f"标签表 {self.label_file!r} 里 {len(wanted)} 个 slide 中，"
                   f"{len(orphans)} 个在**所有**配置染色下都没有特征文件，已显式排除: "
                   f"{orphans}")
            print("!" * 78)
            print("! WARNING — 显式排除标签表孤儿（非静默）")
            print("! " + msg)
            print("!" * 78)

        for slide_id in wanted:
            if slide_id in ref:
                self.samples.append({
                    'slide_id': slide_id,
                    'label': self.labels[slide_id],
                })

        if self.strict_modalities and not self.samples:
            raise RuntimeError(
                f"没有任何样本同时满足模态 {self.modalities} 与标签 "
                f"{self.label_file!r}")

        counts = {}
        for s in self.samples:
            counts[s['label']] = counts.get(s['label'], 0) + 1
        if self.verbose:
            print(f"C17Dataset initialized: {len(self.samples)} samples "
                  f"(标签表 {len(wanted)} − 孤儿 {len(orphans)})")
            print(f"  Label distribution: {counts}")
            print(f"  Modalities ({len(self.modalities)}): {self.modalities}, "
                  f"feature dim will be checked per sample")

    def _preload_features(self):
        for mod in self.modalities:
            self.features_cache[mod] = {}
        for s in self.samples:
            sid = s['slide_id']
            for mod in self.modalities:
                t = torch.load(str(self._feature_path(mod, sid)),
                               map_location='cpu', weights_only=True)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                self.features_cache[mod][sid] = t

    # ── sampling ──────────────────────────────────────────────────────────

    def _build_indices(self, slide_id: str, total_patches: int) -> Optional[np.ndarray]:
        """为一张 slide 生成 patch 索引（只调一次，所有染色共用）。"""
        if not self.max_patches or total_patches <= self.max_patches:
            return None
        if self.sampling == 'first':
            return np.arange(self.max_patches)
        epoch = self._epoch if self.per_epoch else 0
        seed = stable_slide_seed(slide_id, self.sample_seed, epoch)
        rng = np.random.RandomState(seed)
        idx = rng.choice(total_patches, self.max_patches, replace=False)
        idx.sort()
        return idx

    # ── dataset protocol ──────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, mod: str, slide_id: str) -> torch.Tensor:
        feat = torch.load(str(self._feature_path(mod, slide_id)),
                          map_location='cpu', weights_only=True)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        return feat

    def __getitem__(self, idx):
        sample = self.samples[idx]
        slide_id, label = sample['slide_id'], sample['label']

        first_mod = self.modalities[0]
        if self.preload and first_mod in self.features_cache:
            feat_first = self.features_cache[first_mod][slide_id]
        else:
            feat_first = self._load_feature(first_mod, slide_id)
        total_patches = feat_first.shape[0]

        patch_indices = self._build_indices(slide_id, total_patches)

        features = {}
        for i, mod in enumerate(self.modalities):
            if i == 0:
                feat = feat_first
            elif self.preload and mod in self.features_cache:
                feat = self.features_cache[mod][slide_id]
            else:
                feat = self._load_feature(mod, slide_id)

            # 共享 patch 索引的前提：数量与维度必须一致。不静默截断。
            if i > 0 and self.require_same_patch_count:
                if feat.shape[0] != total_patches:
                    raise ValueError(
                        f"{slide_id}: 跨染色 patch 数不一致 — "
                        f"{first_mod}={total_patches}, {mod}={feat.shape[0]}. "
                        f"共享 patch 索引要求五个染色 patch 数完全相同。")
            if feat.shape[-1] != feat_first.shape[-1]:
                raise ValueError(
                    f"{slide_id}: 特征维度不一致 — "
                    f"{first_mod}={feat_first.shape[-1]}, {mod}={feat.shape[-1]}")

            features[mod] = feat[patch_indices] if patch_indices is not None else feat

        return {'features': features, 'label': label, 'slide_id': slide_id}


def c17_multimodal_collate_fn(batch):
    """C17 collate：与 ``c16_multimodal_collate_fn`` 输出结构逐字段一致。"""
    if len(batch) == 0:
        return {'features': [], 'labels': torch.tensor([]), 'slide_ids': []}

    modalities = list(batch[0]['features'].keys())
    features_by_modality = [
        [item['features'][mod] for item in batch] for mod in modalities
    ]
    return {
        'features': features_by_modality,
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'slide_ids': [item['slide_id'] for item in batch],
    }
