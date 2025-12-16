"""
for data augmentation
"""
import numpy as np
import torch

# Normalization
# -> original dataset has already noralize rgb
# class NormalizeFeatures:
#     def __call__(self, points):
#         # Normalize [x, y, z] spatial coordinates -> 
#         points[:, :3] -= np.mean(points[:, :3], axis=0)
#         points[:, :3] /= np.max(np.linalg.norm(points[:, :3], axis=1))
        
#         # Normalize RGB to [0, 1] -> hsv is already [0, 1]
#         points[:, 3:6] /= 255.0
        
#         return points


class NormalizeFeatures:
    def __init__(self):
        self.mean = None
        self.variance = None

    def __call__(self, points):
        # Calculate mean and variance of [x, y, z] spatial coordinates
        self.mean = np.mean(points[:, :3], axis=0)
        self.variance = np.max(np.linalg.norm(points[:, :3] - self.mean, axis=1))
        
        # Normalize [x, y, z] spatial coordinates
        points[:, :3] -= self.mean
        points[:, :3] /= self.variance
        
        # Normalize RGB to [0, 1] (hsv is already [0, 1])
        points[:, 3:6] /= 255.0
        
        return points

    def unnormalize(self, points):
        # Unnormalize [x, y, z] spatial coordinates
        points[:, :3] *= self.variance
        points[:, :3] += self.mean
        # RGB normalization is irreversible (if needed, store the original separately)
        return points
   

## This is the version when loading the training data points (num_points, 9)
## Adjusting vertebrae rgb colors value to lumbar vertebral column [212,188,102]
## add random_rate to only apply adjustment to some points (other remain the same) when training 

class AdjustRGBColor:
    def __init__(self, vertebrae_labels=[1, 2, 3, 4, 5], target_rgb=(212, 188, 102), adjust_strength=0.3, variance=10, randomize_rate=0.5):
        """
        :param adjust_strength: How strongly to adjust towards the target RGB (0 = no change, 1 = full change).
        :param variance: Maximum random variance to add per RGB channel for generalization.
        :param randomize_rate: Proportion of vertebrae points to apply the adjustment to (0 = none, 1 = all).
        """
        self.vertebrae_labels = vertebrae_labels
        self.target_rgb = target_rgb
        self.adjust_strength = adjust_strength
        self.variance = variance
        self.randomize_rate = randomize_rate
    
    def __call__(self, points):
        # Identify vertebrae points
        vertebrae_points = np.isin(points[:, 9], self.vertebrae_labels)
        
        # Create a mask for random selection of vertebrae points
        random_mask = np.random.rand(points.shape[0]) < self.randomize_rate
        points_to_adjust = vertebrae_points & random_mask
        
        # Adjust RGB values for selected points
        current_rgb = points[points_to_adjust, 3:6]
        adjustment = (self.target_rgb - current_rgb) * self.adjust_strength

        # Apply small random variance to each channel
        random_variance = np.random.randint(-self.variance, self.variance + 1, current_rgb.shape)
        adjusted_rgb = current_rgb + adjustment + random_variance

        # Ensure values stay within the [0, 255] range
        points[points_to_adjust, 3:6] = np.clip(adjusted_rgb, 0, 255)
        return points
        


class RandomizeRateScheduler:
    def __init__(self, start_rate=1.0, end_rate=0.0, decay_epochs=10):
        """
        :param start_rate: Initial randomize rate.
        :param end_rate: Minimum randomize rate.
        :param decay_epochs: Number of epochs for each step reduction.
        """
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.decay_epochs = decay_epochs
        self.current_rate = start_rate

    def step(self, epoch):
        """
        Adjust the randomize rate based on the epoch.
        """
        if epoch > 0 and epoch % self.decay_epochs == 0 and self.current_rate > self.end_rate:
            self.current_rate = max(self.end_rate, self.current_rate - 0.1)  # Reduce by 0.1 each step
        return self.current_rate


### This is the version when training the for each epoch, so points (batch_size, num_points, 9)
class ScheduleAdjustRGBColor:
    def __init__(self, vertebrae_labels=[1, 2, 3, 4, 5], target_rgb=(212, 188, 102), adjust_strength=0.3, variance=10):
        """
        :param adjust_strength: How strongly to adjust towards the target RGB (0 = no change, 1 = full change).
        :param variance: Maximum random variance to add per RGB channel for generalization.
        :param randomize_rate: Proportion of vertebrae points to apply the adjustment to (0 = none, 1 = all).
        """
        self.vertebrae_labels = vertebrae_labels
        self.target_rgb = torch.tensor(target_rgb, dtype=torch.float32)
        self.adjust_strength = adjust_strength
        self.variance = variance
        self.randomize_rate = 1.0
    
    def set_randomize_rate(self, rate):
        """
        Update the randomize rate dynamically.
        """
        self.randomize_rate = rate
    
    def __call__(self, points, target):
        """
        Apply random RGB adjustment to vertebrae points based on the target labels.
        :param points: Tensor of shape (B, N, 9) containing the point cloud data (without labels) and already convert to binary.
        :param target: Tensor of shape (B, N) containing the labels for the points.
        :return: Adjusted points tensor of shape (B, N, 9).
        """
        batch_size, num_points, _ = points.shape


        # Create a mask for vertebrae points based on the target labels
        vertebrae_mask = torch.isin(target, torch.tensor(self.vertebrae_labels))

        # Generate a random mask for selective adjustment
        random_mask = torch.rand((batch_size, num_points)) < self.randomize_rate
        adjust_mask = vertebrae_mask & random_mask

        # Adjust RGB values for the selected points
        current_rgb = points[..., 3:6]  # RGB channels
        adjustment = (self.target_rgb - current_rgb) * self.adjust_strength
        random_variance = torch.randint(
            -self.variance, self.variance + 1, current_rgb.shape
        )
        adjusted_rgb = current_rgb + adjustment + random_variance

        # Clip RGB values and apply only to the selected points
        adjusted_rgb = torch.clip(adjusted_rgb, 0, 255)
        points[..., 3:6] = torch.where(adjust_mask.unsqueeze(-1), adjusted_rgb, current_rgb)
        # print(f"before normalizing: {points}")
#### Add Normalizing since 
        # Normalize [x, y, z] spatial coordinates
        points[..., :3] -= torch.mean(points[..., :3], dim=0)
        norms = torch.norm(points[..., :3], dim=1)
        max_norm = torch.max(norms)
        epsilon = 1e-6  # Small constant to prevent division by zero
        if max_norm > epsilon:
            points[..., :3] /= max_norm

        # Normalize RGB to [0, 1]
        points[..., 3:6] /= 255.0
        
        return points
    
        
    
        