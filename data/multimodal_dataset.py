"""
MultiModalFeatureDataset: 多模态特征数据集

支持从多个模态目录加载特征，每个样本包含所有模态的特征。

数据目录结构:
    features_result/C17_features/
        C17_raw_features/
            patient_001/
                node_1.pt
                node_2.pt
        C17_ER_features/
            patient_001/
                node_1.pt
                node_2.pt
        ... (其他模态)

标签文件格式 (CSV):
    slide_id,label
    patient_001_node_1,0
    patient_001_node_2,1
    ...
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union


class MultiModalFeatureDataset(Dataset):
    """
    多模态特征数据集
    
    每个样本包含多个模态的特征，模态通过列表指定。
    """
    def __init__(self, 
                 feature_dirs: Dict[str, str],
                 label_file: str,
                 patient_list: Optional[List[str]] = None,
                 max_patches: Optional[int] = None,
                 preload: bool = False,
                 label_level: str = 'slide',
                 align_patches: bool = True,
                 verbose: bool = True):
        """
        Args:
            feature_dirs: 模态名称到特征目录的映射
                e.g., {'RAW': '/path/to/C17_raw_features', 
                       'ER': '/path/to/C17_ER_features'}
            label_file: CSV标签文件路径
            patient_list: 指定的患者列表
            max_patches: 每个WSI最大patch数量
            preload: 是否预加载所有特征到内存
            label_level: 'slide' 或 'patient'
            align_patches: 是否对齐不同模态的patch (确保每个模态的patch数量一致)
            verbose: 是否打印详细信息
        """
        self.feature_dirs = {k: Path(v) for k, v in feature_dirs.items()}
        self.modalities = list(feature_dirs.keys())
        self.num_modalities = len(self.modalities)
        self.max_patches = max_patches
        self.preload = preload
        self.label_level = label_level
        self.align_patches = align_patches
        self.verbose = verbose
        
        if verbose:
            print(f"Initializing MultiModalFeatureDataset with {self.num_modalities} modalities: {self.modalities}")
        
        # 加载标签
        self.labels = {}
        self._load_labels(label_file)
        
        # 构建样本列表
        self.samples = []
        self._build_samples(patient_list)
        
        # 预加载
        if self.preload:
            self.features_cache = {}
            self._preload_features()
    
    def _load_labels(self, label_file):
        """加载标签文件"""
        if label_file is None or not os.path.exists(label_file):
            print(f"Warning: Label file not found: {label_file}")
            return
        
        df = pd.read_csv(label_file)
        
        if 'slide_id' in df.columns:
            self.label_level = 'slide'
            for _, row in df.iterrows():
                self.labels[str(row['slide_id'])] = int(row['label'])
            if self.verbose:
                print(f"Loaded {len(self.labels)} slide-level labels")
        elif 'patient_id' in df.columns:
            self.label_level = 'patient'
            for _, row in df.iterrows():
                self.labels[str(row['patient_id'])] = int(row['label'])
            if self.verbose:
                print(f"Loaded {len(self.labels)} patient-level labels")
        else:
            raise ValueError("Label file must have 'slide_id' or 'patient_id' column")
    
    def _build_samples(self, patient_list):
        """构建样本列表"""
        # 获取第一个模态的患者列表
        first_modality = self.modalities[0]
        first_dir = self.feature_dirs[first_modality]
        
        if patient_list is not None:
            patients = patient_list
        else:
            patients = sorted([d.name for d in first_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('patient_')])
        
        # 对于每个患者，找到所有slide/node
        for patient_id in patients:
            # 获取第一个模态的所有pt文件
            first_patient_path = first_dir / patient_id
            if not first_patient_path.exists():
                continue
            
            pt_files = sorted(first_patient_path.glob('*.pt'))
            
            for pt_file in pt_files:
                slide_id = pt_file.stem
                full_slide_id = f"{patient_id}_{slide_id}"
                
                # 检查所有模态是否都有这个slide
                modality_files = {}
                all_exist = True
                
                for modality in self.modalities:
                    mod_file = self.feature_dirs[modality] / patient_id / f"{slide_id}.pt"
                    if mod_file.exists():
                        modality_files[modality] = mod_file
                    else:
                        all_exist = False
                        if self.verbose:
                            print(f"Warning: {modality} feature not found for {full_slide_id}")
                        break
                
                if not all_exist:
                    continue
                
                # 获取标签
                if self.label_level == 'slide':
                    label = self.labels.get(full_slide_id, 
                                          self.labels.get(slide_id, -1))
                else:
                    label = self.labels.get(patient_id, -1)
                
                if label == -1:
                    if self.verbose:
                        print(f"Warning: No label found for {full_slide_id}")
                    continue
                
                self.samples.append({
                    'patient_id': patient_id,
                    'slide_id': slide_id,
                    'full_slide_id': full_slide_id,
                    'modality_files': modality_files,
                    'label': label
                })
        
        if self.verbose:
            print(f"Dataset initialized: {len(self.samples)} samples from {len(patients)} patients")
    
    def _preload_features(self):
        """预加载所有特征到内存"""
        if self.verbose:
            print("Preloading features to memory...")
        
        for sample in self.samples:
            self.features_cache[sample['full_slide_id']] = {}
            for modality, file_path in sample['modality_files'].items():
                self.features_cache[sample['full_slide_id']][modality] = torch.load(file_path)
        
        if self.verbose:
            print("Preloading complete!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载所有模态的特征
        features_list = []
        
        for modality in self.modalities:
            if self.preload:
                features = self.features_cache[sample['full_slide_id']][modality]
            else:
                features = torch.load(sample['modality_files'][modality])
            
            # 限制patch数量
            if self.max_patches is not None and self.max_patches > 0:
                if len(features) > self.max_patches:
                    indices = torch.randperm(len(features))[:self.max_patches]
                    features = features[indices]
            
            features_list.append(features)
        
        # 对齐不同模态的patch数量
        if self.align_patches:
            features_list = self._align_patches(features_list)
        
        label = sample['label']
        
        return {
            'features': features_list,  # list of [N, D] with length num_modalities
            'label': label,
            'patient_id': sample['patient_id'],
            'slide_id': sample['slide_id'],
            'full_slide_id': sample['full_slide_id']
        }
    
    def _align_patches(self, features_list):
        """对齐不同模态的patch数量"""
        # 找到最小的patch数量
        min_patches = min([len(f) for f in features_list])
        
        aligned = []
        for features in features_list:
            if len(features) > min_patches:
                # 随机采样
                indices = torch.randperm(len(features))[:min_patches]
                features = features[indices]
            aligned.append(features)
        
        return aligned
    
    def get_modality_names(self):
        """获取模态名称列表"""
        return self.modalities
    
    def get_label_distribution(self):
        """获取标签分布"""
        labels = [s['label'] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


def create_multimodal_datasets(feature_dirs, label_file,
                               train_patients=None, val_patients=None, test_patients=None,
                               max_patches=None, preload=False, **kwargs):
    """
    创建训练、验证、测试多模态数据集
    
    Args:
        feature_dirs: 模态名称到特征目录的映射
        label_file: 标签文件路径
        train_patients: 训练集患者列表
        val_patients: 验证集患者列表
        test_patients: 测试集患者列表
        max_patches: 最大patch数
        preload: 是否预加载
        **kwargs: 其他参数
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_dataset = MultiModalFeatureDataset(
        feature_dirs=feature_dirs,
        label_file=label_file,
        patient_list=train_patients,
        max_patches=max_patches,
        preload=preload,
        **kwargs
    ) if train_patients else None
    
    val_dataset = MultiModalFeatureDataset(
        feature_dirs=feature_dirs,
        label_file=label_file,
        patient_list=val_patients,
        max_patches=max_patches,
        preload=preload,
        **kwargs
    ) if val_patients else None
    
    test_dataset = MultiModalFeatureDataset(
        feature_dirs=feature_dirs,
        label_file=label_file,
        patient_list=test_patients,
        max_patches=max_patches,
        preload=preload,
        **kwargs
    ) if test_patients else None
    
    return train_dataset, val_dataset, test_dataset


def multimodal_collate_fn(batch):
    """
    多模态数据的collate function
    
    每个batch包含多个模态的特征列表
    """
    if isinstance(batch[0], dict):
        # 收集所有模态的特征
        num_modalities = len(batch[0]['features'])
        
        # features_by_modality[i] 是第i个模态的所有样本的特征列表
        features_by_modality = [[] for _ in range(num_modalities)]
        labels = []
        patient_ids = []
        slide_ids = []
        
        for item in batch:
            for i in range(num_modalities):
                features_by_modality[i].append(item['features'][i])
            labels.append(item['label'])
            patient_ids.append(item.get('patient_id', ''))
            slide_ids.append(item.get('slide_id', ''))
        
        return {
            'features': features_by_modality,  # list of list of tensors
            'labels': torch.tensor(labels),
            'patient_ids': patient_ids,
            'slide_ids': slide_ids
        }
    else:
        # 简单tuple格式不支持多模态
        raise ValueError("MultiModalFeatureDataset should return dict")


# 便捷函数：创建常见模态组合的数据集
def create_dual_modal_dataset(base_dir, modality1='RAW', modality2='ER', **kwargs):
    """创建双模态数据集 (RAW + 另一个模态)"""
    feature_dirs = {
        modality1: f"{base_dir}/C17_raw_features",
        modality2: f"{base_dir}/C17_{modality2.lower()}_features"
    }
    return MultiModalFeatureDataset(feature_dirs=feature_dirs, **kwargs)


def create_triple_modal_dataset(base_dir, modalities=['RAW', 'ER', 'PR'], **kwargs):
    """创建三模态数据集"""
    feature_dirs = {}
    for mod in modalities:
        if mod == 'RAW':
            feature_dirs[mod] = f"{base_dir}/C17_raw_features"
        else:
            feature_dirs[mod] = f"{base_dir}/C17_{mod.lower()}_features"
    return MultiModalFeatureDataset(feature_dirs=feature_dirs, **kwargs)


def create_five_modal_dataset(base_dir, **kwargs):
    """创建五模态数据集 (RAW, ER, PR, HER2, KI67)"""
    feature_dirs = {
        'RAW': f"{base_dir}/C17_raw_features",
        'ER': f"{base_dir}/C17_ER_features",
        'PR': f"{base_dir}/C17_PR_features",
        'HER2': f"{base_dir}/C17_HER2_features",
        'KI67': f"{base_dir}/C17_KI67_features"
    }
    return MultiModalFeatureDataset(feature_dirs=feature_dirs, **kwargs)


if __name__ == "__main__":
    print("Testing MultiModalFeatureDataset...")
    
    # 创建模拟数据目录结构 (仅用于测试)
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建模拟目录
        raw_dir = Path(temp_dir) / "raw_features"
        er_dir = Path(temp_dir) / "er_features"
        
        for d in [raw_dir, er_dir]:
            patient_dir = d / "patient_001"
            patient_dir.mkdir(parents=True)
            # 创建模拟特征文件
            torch.save(torch.randn(100, 768), patient_dir / "node_1.pt")
            torch.save(torch.randn(80, 768), patient_dir / "node_2.pt")
        
        # 创建模拟标签文件
        label_file = Path(temp_dir) / "labels.csv"
        with open(label_file, 'w') as f:
            f.write("slide_id,label\n")
            f.write("patient_001_node_1,0\n")
            f.write("patient_001_node_2,1\n")
        
        # 测试数据集
        feature_dirs = {
            'RAW': str(raw_dir),
            'ER': str(er_dir)
        }
        
        dataset = MultiModalFeatureDataset(
            feature_dirs=feature_dirs,
            label_file=str(label_file),
            max_patches=50,
            verbose=True
        )
        
        print(f"\nDataset length: {len(dataset)}")
        
        # 测试获取数据
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Number of modalities: {len(sample['features'])}")
        print(f"Feature shapes: {[f.shape for f in sample['features']]}")
        print(f"Label: {sample['label']}")
        
        # 测试collate_fn
        batch = [dataset[0], dataset[1]]
        collated = multimodal_collate_fn(batch)
        print(f"\nCollated batch keys: {collated.keys()}")
        print(f"Number of modalities in batch: {len(collated['features'])}")
        print(f"Labels: {collated['labels']}")
        
        print("\nAll tests passed!")
        
    finally:
        # 清理
        shutil.rmtree(temp_dir)
