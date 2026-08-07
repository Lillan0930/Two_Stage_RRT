"""
C16 Multimodal Feature Dataset

支持 C16 数据结构：{feature_dir}/{normal,tumor,test}/file.pt
每个模态下有 normal/ tumor/ test/ 三个子目录，含 .pt 特征文件。
"""

import os
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


class C16MultimodalDataset:
    """
    C16 多模态特征数据集。

    特征目录结构:
        C16_HE_features/
            normal/normal_001.pt ... normal_160.pt
            tumor/tumor_001.pt ... tumor_111.pt
            test/test_001.pt ... test_129.pt
    """

    def __init__(self,
                 feature_dirs: Dict[str, str],
                 label_file: str,
                 max_patches: Optional[int] = None,
                 preload: bool = False,
                 verbose: bool = True):
        self.feature_dirs = {k: Path(v) for k, v in feature_dirs.items()}
        self.modalities = list(feature_dirs.keys())
        self.num_modalities = len(self.modalities)
        self.max_patches = max_patches
        self.preload = preload
        self.verbose = verbose

        if verbose:
            print(f"Initializing C16MultimodalDataset with {self.num_modalities} modalities: {self.modalities}")

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

    def _load_labels(self, label_file):
        """加载 label CSV (slide_id,label)"""
        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            self.labels[str(row['slide_id'])] = int(row['label'])
        if self.verbose:
            print(f"Loaded {len(self.labels)} labels")

    def _build_samples(self):
        """构建样本列表，匹配所有模态的特征文件"""
        # 用第一个模态的文件作为基准
        first_mod = self.modalities[0]
        first_dir = self.feature_dirs[first_mod]

        # 收集所有 .pt 文件的 slide_id
        slide_feature_map = {}
        for subdir in ['normal', 'tumor', 'test']:
            subpath = first_dir / subdir
            if not subpath.exists():
                continue
            for f in subpath.glob('*.pt'):
                slide_id = f.stem  # e.g., "tumor_078", "normal_001", "test_011"
                if slide_id in self.labels:
                    slide_feature_map[slide_id] = f

        # 检查其他模态是否都有对应文件
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

        # 构建样本
        for slide_id in sorted(slide_feature_map.keys()):
            self.samples.append({
                'slide_id': slide_id,
                'label': self.labels[slide_id],
            })

        # 统计
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
        mod_dir = self.feature_dirs[mod]
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

        # 加载各模态特征
        features = {}
        for mod in self.modalities:
            if self.preload and mod in self.features_cache:
                feat = self.features_cache[mod][slide_id]
            else:
                mod_dir = self.feature_dirs[mod]
                feat = None
                for subdir in ['normal', 'tumor', 'test']:
                    candidate = mod_dir / subdir / f"{slide_id}.pt"
                    if candidate.exists():
                        feat = torch.load(str(candidate), map_location='cpu', weights_only=True)
                        if feat.dim() == 1:
                            feat = feat.unsqueeze(0)
                        break
                if feat is None:
                    raise FileNotFoundError(f"Feature not found for {slide_id} in {mod}")

            # 截断到 max_patches
            if self.max_patches and feat.shape[0] > self.max_patches:
                feat = feat[:self.max_patches]
            features[mod] = feat

        return {
            'features': features,
            'label': label,
            'slide_id': slide_id,
        }


def c16_multimodal_collate_fn(batch):
    """C16 collate: features by modality → list of tensors per sample"""
    if len(batch) == 0:
        return {'features': [], 'labels': torch.tensor([]), 'slide_ids': []}

    modalities = list(batch[0]['features'].keys())

    # features: list of list — [modality][sample] = tensor
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
