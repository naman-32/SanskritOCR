# Lightweight Image Preprocessing Utility for Low-Resolution images
# Data augmentation utility using noise, blur, skew, transformation
# Handles degraded manuscript scans with augmentation and super-resolution.
# Author: Naman Goenka
# Date: Dec 25th 2020

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Preprocessor for handling degraded manuscript scans.
    Includes data augmentation and super-resolution capabilities.
    """
    
    def __init__(self, enable_super_resolution=True):
        """
        Initialize the image preprocessor.
        
        Args:
            enable_super_resolution: Whether to apply super-resolution
        """
        self.enable_super_resolution = enable_super_resolution
    
    def add_gaussian_noise(self, image: np.ndarray, mean=0, sigma=10) -> np.ndarray:
        """
        Add Gaussian noise to simulate degraded manuscripts.
        
        Args:
            image: Input image
            mean: Mean of Gaussian distribution
            sigma: Standard deviation of Gaussian distribution
        
        Returns:
            Noisy image
        """
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    def add_salt_pepper_noise(self, image: np.ndarray, amount=0.01) -> np.ndarray:
        """
        Add salt and pepper noise to simulate ink degradation.
        
        Args:
            image: Input image
            amount: Proportion of pixels to add noise to
        
        Returns:
            Noisy image
        """
        noisy = image.copy()
        
        # Salt noise (white pixels)
        num_salt = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy[coords[0], coords[1]] = 255
        
        # Pepper noise (black pixels)
        num_pepper = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy[coords[0], coords[1]] = 0
        
        return noisy
    
    def apply_blur(self, image: np.ndarray, kernel_size=5) -> np.ndarray:
        """
        Apply Gaussian blur to simulate focus issues.
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
        
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_motion_blur(self, image: np.ndarray, size=15) -> np.ndarray:
        """
        Apply motion blur to simulate camera shake.
        
        Args:
            image: Input image
            size: Size of motion blur kernel
        
        Returns:
            Motion blurred image
        """
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(image, -1, kernel)
    
    def apply_skew(self, image: np.ndarray, angle=5.0) -> np.ndarray:
        """
        Apply skew transformation to simulate scanning angle issues.
        
        Args:
            image: Input image
            angle: Skew angle in degrees
        
        Returns:
            Skewed image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        skewed = cv2.warpAffine(image, M, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        return skewed
    
    def apply_perspective_transform(self, image: np.ndarray, 
                                   distortion=0.1) -> np.ndarray:
        """
        Apply perspective transformation to simulate page curvature.
        
        Args:
            image: Input image
            distortion: Amount of distortion (0-1)
        
        Returns:
            Transformed image
        """
        h, w = image.shape[:2]
        
        # Define source points (corners of original image)
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Define destination points with distortion
        offset = int(w * distortion)
        dst_points = np.float32([
            [offset, 0], 
            [w - offset, 0], 
            [0, h], 
            [w, h]
        ])
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        transformed = cv2.warpPerspective(image, M, (w, h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(255, 255, 255))
        return transformed
    
    def denoise(self, image: np.ndarray, strength=10) -> np.ndarray:
        """
        Apply denoising to clean up degraded images.
        
        Args:
            image: Input image
            strength: Denoising strength
        
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image to enhance text clarity.
        
        Args:
            image: Input image
        
        Returns:
            Sharpened image
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for better binarization.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Binarized image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    def super_resolution(self, image: np.ndarray, scale=2) -> np.ndarray:
        """
        Apply lightweight super-resolution to enhance image quality.
        Uses bicubic interpolation with sharpening.
        
        Args:
            image: Input image
            scale: Upscaling factor
        
        Returns:
            Super-resolved image
        """
        h, w = image.shape[:2]
        
        # Upscale using bicubic interpolation
        upscaled = cv2.resize(image, (w * scale, h * scale), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Apply unsharp masking for enhancement
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        enhanced = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
        
        return enhanced
    
    def preprocess(self, image: np.ndarray, 
                   denoise_strength=10,
                   apply_sr=None) -> np.ndarray:
        """
        Complete preprocessing pipeline for degraded manuscripts.
        
        Args:
            image: Input image
            denoise_strength: Strength of denoising (0 to disable)
            apply_sr: Whether to apply super-resolution (None uses default)
        
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Denoise if requested
        if denoise_strength > 0:
            processed = self.denoise(processed, denoise_strength)
        
        # Super-resolution if enabled
        if apply_sr is None:
            apply_sr = self.enable_super_resolution
        
        if apply_sr:
            processed = self.super_resolution(processed, scale=2)
        
        # Sharpen
        processed = self.sharpen(processed)
        
        return processed
    
    def augment_for_training(self, image: np.ndarray, 
                            augmentation_type='all') -> list:
        """
        Generate augmented versions for training data.
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation ('all', 'noise', 'blur', 'geometric')
        
        Returns:
            List of augmented images
        """
        augmented = [image.copy()]  # Original
        
        if augmentation_type in ['all', 'noise']:
            # Noise variations
            augmented.append(self.add_gaussian_noise(image, sigma=5))
            augmented.append(self.add_gaussian_noise(image, sigma=15))
            augmented.append(self.add_salt_pepper_noise(image, amount=0.005))
            augmented.append(self.add_salt_pepper_noise(image, amount=0.02))
        
        if augmentation_type in ['all', 'blur']:
            # Blur variations
            augmented.append(self.apply_blur(image, kernel_size=3))
            augmented.append(self.apply_blur(image, kernel_size=7))
            augmented.append(self.apply_motion_blur(image, size=10))
        
        if augmentation_type in ['all', 'geometric']:
            # Geometric variations
            augmented.append(self.apply_skew(image, angle=3))
            augmented.append(self.apply_skew(image, angle=-3))
            augmented.append(self.apply_perspective_transform(image, distortion=0.05))
        
        return augmented


# Utility functions for easy integration
def preprocess_image(image_path: str, 
                    enable_super_resolution=True,
                    denoise_strength=10) -> np.ndarray:
    """
    Convenience function to preprocess a single image.
    
    Args:
        image_path: Path to input image
        enable_super_resolution: Whether to apply super-resolution
        denoise_strength: Denoising strength
    
    Returns:
        Preprocessed image
    """
    image = cv2.imread(image_path)
    preprocessor = ImagePreprocessor(enable_super_resolution)
    return preprocessor.preprocess(image, denoise_strength)


def augment_image(image_path: str, 
                 output_dir: str,
                 augmentation_type='all') -> int:
    """
    Convenience function to generate augmented versions of an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save augmented images
        augmentation_type: Type of augmentation
    
    Returns:
        Number of augmented images generated
    """
    import os
    
    image = cv2.imread(image_path)
    preprocessor = ImagePreprocessor()
    augmented = preprocessor.augment_for_training(image, augmentation_type)
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for i, aug_img in enumerate(augmented):
        output_path = os.path.join(output_dir, f"{base_name}_aug{i}.png")
        cv2.imwrite(output_path, aug_img)
    
    return len(augmented)


if __name__ == "__main__":
    # Example usage
    print("Image Preprocessor Module")
    print("Supports: denoising, super-resolution, augmentation")
    print("Use: from utils.image_preprocessor import ImagePreprocessor")
