import numpy as np
import os
import torch, random
from torch.utils.data import Dataset
import re

"""
Create separate dataset to make train data more flexible
For test data (evaluation after each epoch) reproducible (always evaluate on the same test dataset)
ptv3_6 version is for 6 classes 0, 1-5
"""

## For Training
class TrainDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, test_specimen_idx=9, sample_ratio = 0.2, num_examples=25, transforms=None):
        """
        Training dataset: Random sampling to create batches for training.
        """
        self.num_points = num_points
        self.sample_ratio = sample_ratio
        self.transforms = transforms
        self.root_dir = root_dir
        self.num_examples = num_examples

        # Retrieve training files, excluding the test specimen
        self.files = self._get_files(root_dir, exclude_specimen_idx=test_specimen_idx)

        # Balance dataset by limiting the number of examples per specimen
        self.files = self._balance_specimen_files(self.files, num_examples)

        self.labelweights = self.calculate_label_weights()

    def _get_files(self, root_dir, exclude_specimen_idx):
        """Retrieve training files, excluding the test specimen."""
        all_files = sorted(f for f in os.listdir(root_dir) if f.endswith('.npy'))
        return [f for f in all_files if f'S_{exclude_specimen_idx}' not in f]
    
    def _balance_specimen_files(self, files, num_examples):
        """Ensures each specimen has exactly num_examples files."""
        specimen_files = {}

        # Group files by specimen index
        for f in files:
            match = re.search(r'S_(\d+)_V', f)  # Extracts specimen index
            if match:
                specimen_idx = int(match.group(1))
                specimen_files.setdefault(specimen_idx, []).append(f)
            else:
                print(f"Warning: Could not parse specimen index from {f}")

        print("Before balancing:", {k: len(v) for k, v in specimen_files.items()})  # Debugging

        balanced_files = []
        for specimen_idx, file_list in specimen_files.items():
            if len(file_list) >= num_examples:
                selected_files = random.sample(file_list, num_examples)
            else:
                selected_files = file_list  

            balanced_files.extend(selected_files)

        print("After balancing:", {k: len(v) for k, v in specimen_files.items()})  # Debugging
        return balanced_files
    

    def calculate_label_weights(self):
        """
        Calculate label weights to handle class imbalance.
        label weigths are 'Before' sampling
        apply in loss function
        """
        total_labels = np.zeros(6)  # 6 labels: 0 and 1-5
        for file_name in self.files:
            file_path = os.path.join(self.root_dir, file_name)
            specimen_data = np.load(file_path)
            labels = specimen_data[:, 9]

            # Count occurrences of each label
            tmp, _ = np.histogram(labels, bins=np.arange(7))
            total_labels += tmp

        # Normalize label weights
        total_labels = total_labels.astype(np.float32) / np.sum(total_labels)
        
        # Inversely proportional weights
        label_weights = np.power(np.max(total_labels) / total_labels, 1 / 3.0)
        print(f"Label weights (Train): {label_weights}")
        return label_weights
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        points = np.load(file_path)

        labels = points[:, 9]

        # Separate class 0 and class 1 points
        class_0 = points[labels == 0]
        class_1_5 = points[labels > 0]

        #  Set a target ratio
        sample_ratio = self.sample_ratio
        num_class_1_5 = min(int(self.num_points * sample_ratio), len(class_1_5))
        num_class_0 = self.num_points - num_class_1_5

        class_0_sample = class_0[np.random.choice(len(class_0), num_class_0, replace=len(class_0) < num_class_0)]
        class_1_5_sample = class_1_5[np.random.choice(len(class_1_5), num_class_1_5, replace=len(class_1_5) < num_class_1_5)]

        sampled_points = np.concatenate([class_0_sample, class_1_5_sample], axis=0)
        np.random.shuffle(sampled_points)
        
        # Apply transformations
        if self.transforms:
            sampled_points = self.transforms(sampled_points)

        features = sampled_points[:, :9]  # [x, y, z, r, g, b, h, s, v]
        labels = sampled_points[:, 9]


        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


## For Evaluation -> always the same data
class TestDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, test_specimen_idx=9, seed=42, transforms=None):
        """
        Testing dataset: Fixed sampling for reproducibility.
        Set seed for np.random.seed
        """
        self.num_points = num_points
        self.files = self._get_files(root_dir, include_specimen_idx=test_specimen_idx)
        self.root_dir = root_dir
        self.seed = seed
        self.transforms = transforms
        
        self.labelweights = self.calculate_label_weights()
        
    def _get_files(self, root_dir, include_specimen_idx):
        """Retrieve testing files, only including the test specimen."""
        all_files = sorted(f for f in os.listdir(root_dir) if f.endswith('.npy'))
        return [f for f in all_files if f'S_{include_specimen_idx}' in f]
    
    def calculate_label_weights(self):
        """
        Calculate label weights to handle class imbalance.
        label weigths are 'Before' sampling
        apply in loss function
        """
        total_labels = np.zeros(6)  # 6 labels: 0 and 1-5
        for file_name in self.files:
            file_path = os.path.join(self.root_dir, file_name)
            specimen_data = np.load(file_path)
            labels = specimen_data[:, 9]

            # Count occurrences of each label
            tmp, _ = np.histogram(labels, bins=np.arange(7))
            total_labels += tmp

        # Normalize label weights
        total_labels = total_labels.astype(np.float32) / np.sum(total_labels)
        
        # Inversely proportional weights
        label_weights = np.power(np.max(total_labels) / total_labels, 1 / 3.0)
        print(f"Label weights (Test): {label_weights}")
        return label_weights
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        np.random.seed(self.seed)  # Set fixed seed for reproducibility
        file_path = os.path.join(self.root_dir, self.files[idx])
        points = np.load(file_path)

        labels = points[:, 9]

        # Separate class 0 and class 1 points
        class_0 = points[labels == 0]
        class_1_5 = points[labels > 0]

        # Sample points with an 8:2 ratio
        num_class_1_5 = min(int(self.num_points * 0.2), len(class_1_5))
        num_class_0 = self.num_points - num_class_1_5

        class_0_sample = class_0[np.random.choice(len(class_0), num_class_0, replace=False)]
        class_1_5_sample = class_1_5[np.random.choice(len(class_1_5), num_class_1_5, replace=False)]

        sampled_points = np.concatenate([class_0_sample, class_1_5_sample], axis=0)
         
        # Apply transformations
        if self.transforms:
            sampled_points = self.transforms(sampled_points)

        features = sampled_points[:, :9]  # [x, y, z, r, g, b, h, s, v]
        labels = sampled_points[:, 9]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
