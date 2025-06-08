import cv2
import numpy as np
from tqdm import tqdm
import os
from utils import VideoProcessor
import hashlib
import json

class BackgroundSubtractor(VideoProcessor):
    def __init__(self):
        super().__init__()
        
        # ========== ALL CONFIGURABLE PARAMETERS ==========
        
        # MOG2 Background Subtraction Parameters
        self.bg_history = 200  # Number of frames for background model
        self.bg_threshold = 20  # Variance threshold for background/foreground classification
        self.detect_shadows = True  # Enable shadow detection to avoid including shadows
        self.learning_rate = 0.01  # Background model learning rate (slower = more conservative)
        
        # Motion Detection Parameters
        self.motion_threshold = 20  # Threshold for frame differencing motion detection
        
        # Median Background Parameters
        self.median_sample_interval = 10  # Sample every Nth frame for median background
        self.median_threshold = 35  # Threshold for median background difference
        
        # Mask Combination Parameters
        self.gmm_weight = 0.6  # Weight for GMM mask in combination
        self.motion_weight = 0.3  # Weight for motion mask in combination
        self.median_weight = 0.1  # Weight for median mask in combination
        self.combined_mask_threshold = 128  # Threshold for final combined mask
        
        # Person Detection Constraints
        self.min_contour_area = 1000  # Minimum area for person detection
        self.max_contour_area = 80000  # Maximum area to filter out large false positives
        self.min_aspect_ratio = 1.2  # Minimum height/width ratio for person
        self.max_aspect_ratio = 4.5  # Maximum height/width ratio for person
        self.min_solidity = 0.3  # Minimum solidity (filled-ness) for person contour
        
        # Morphological Operations Parameters
        self.morph_kernel_size = 3  # Kernel size for initial morphological operations
        self.small_noise_kernel_size = 3  # Kernel size for removing small noise
        self.edge_dilation_size = 3  # Kernel size for edge dilation
        
        # Edge and Smoothing Parameters
        self.gaussian_blur_size = 3  # Kernel size for Gaussian blur smoothing
        self.final_threshold = 127  # Final threshold after Gaussian blur
        
        # Canny Edge Detection Parameters (if using edge refinement)
        self.canny_low_threshold = 50  # Lower threshold for Canny edge detection
        self.canny_high_threshold = 150  # Upper threshold for Canny edge detection
        self.edge_dilation_iterations = 1  # Number of dilation iterations for edges
        
        # ================================================
        
    def extract_person(self, input_path: str, extracted_path: str, binary_path: str):
        """Extract person using optimized background subtraction"""       
        # Read video
        frames, metadata = self.read_video(input_path)
        num_frames = len(frames)
        
        print(f"Processing {num_frames} frames for background subtraction")
        
        # Use multi-method approach for better results
        extracted_frames, binary_frames = self._extract_with_multi_method(frames)
        
        # Save results
        self.write_video(extracted_frames, extracted_path, metadata['fps'])
        self.write_video(binary_frames, binary_path, metadata['fps'])
        
        print(f"Background subtraction complete. Saved {len(extracted_frames)} frames.")
    
    def _extract_with_multi_method(self, frames):
        """Extract using multiple methods combined for better results"""
        print("Using multi-method approach for optimal person extraction...")
        
        # Method 1: MOG2 with optimized parameters
        gmm_masks = self._get_gmm_masks(frames)
        
        # Method 2: Frame differencing for motion detection
        motion_masks = self._get_motion_masks(frames)
        
        # Method 3: Temporal median for static background detection
        median_masks = self._get_median_masks(frames)
        
        extracted_frames = []
        binary_frames = []
        
        print("Combining methods and refining masks...")
        
        for i, frame in enumerate(tqdm(frames, desc="Combining & Refining")):
            # Combine masks using weighted approach
            combined_mask = self._combine_masks(
                gmm_masks[i], 
                motion_masks[i], 
                median_masks[i]
            )
            
            # Refine the mask specifically for person detection
            refined_mask = self._refine_person_mask(combined_mask, frame)
            
            # Extract person using refined mask
            extracted_frame = self._apply_mask_to_frame(frame, refined_mask)
            
            extracted_frames.append(extracted_frame)
            binary_frames.append(refined_mask)
        
        return extracted_frames, binary_frames
    
    def _get_gmm_masks(self, frames):
        """Get masks using GMM method with optimized parameters"""
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history,
            varThreshold=self.bg_threshold,
            detectShadows=self.detect_shadows
        )
        
        masks = []
        for frame in frames:
            mask = bg_subtractor.apply(frame, learningRate=self.learning_rate)
            masks.append(mask)
        
        return masks
    
    def _get_motion_masks(self, frames):
        """Get masks using frame differencing for motion detection"""
        masks = []
        
        for i in range(len(frames)):
            if i == 0:
                # First frame - no motion
                masks.append(np.zeros(frames[0].shape[:2], dtype=np.uint8))
                continue
            
            # Calculate difference between consecutive frames
            diff = cv2.absdiff(frames[i-1], frames[i])
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, motion_mask = cv2.threshold(diff_gray, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up motion mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            
            masks.append(motion_mask)
        
        return masks
    
    def _get_median_masks(self, frames):
        """Get masks using temporal median background"""
        # Compute median background (sample every Nth frame for speed)
        sample_indices = range(0, len(frames), self.median_sample_interval)
        sample_frames = [frames[i] for i in sample_indices]
        sample_stack = np.stack(sample_frames, axis=0).astype(np.float32)
        background = np.median(sample_stack, axis=0).astype(np.uint8)
        
        masks = []
        for frame in frames:
            # Calculate difference from median background
            diff = cv2.absdiff(frame, background)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, median_mask = cv2.threshold(diff_gray, self.median_threshold, 255, cv2.THRESH_BINARY)
            
            masks.append(median_mask)
        
        return masks
    
    def _combine_masks(self, gmm_mask, motion_mask, median_mask):
        """Combine multiple masks using weighted voting"""
        # Convert all masks to same format
        if len(gmm_mask.shape) == 3:
            gmm_mask = cv2.cvtColor(gmm_mask, cv2.COLOR_BGR2GRAY)
        if len(motion_mask.shape) == 3:
            motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_BGR2GRAY)
        if len(median_mask.shape) == 3:
            median_mask = cv2.cvtColor(median_mask, cv2.COLOR_BGR2GRAY)
        
        # Weighted combination using configurable weights
        combined = (self.gmm_weight * gmm_mask.astype(np.float32) + 
                   self.motion_weight * motion_mask.astype(np.float32) + 
                   self.median_weight * median_mask.astype(np.float32))
        
        # Threshold the combined mask
        _, combined_mask = cv2.threshold(combined.astype(np.uint8), self.combined_mask_threshold, 255, cv2.THRESH_BINARY)
        
        return combined_mask
    
    def _refine_person_mask(self, mask, frame):
        """Refine mask specifically for person detection"""
        # Step 1: Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (self.small_noise_kernel_size, self.small_noise_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Step 2: Find contours and filter by size and shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create new mask with only person-like contours
        refined_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area using configurable parameters
            if self.min_contour_area <= area <= self.max_contour_area:
                # Check aspect ratio (person should be taller than wide)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Person typically has aspect ratio between configured bounds
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    # Check solidity (how "filled" the contour is)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Person should have reasonable solidity (not too fragmented)
                    if solidity > self.min_solidity:
                        cv2.fillPoly(refined_mask, [contour], 255)
        
        # Step 3: Smooth edges while preserving details
        if np.any(refined_mask):
            # Slight dilation to ensure we don't lose person edges
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                   (self.edge_dilation_size, self.edge_dilation_size))
            refined_mask = cv2.dilate(refined_mask, kernel_edge, iterations=self.edge_dilation_iterations)
            
            # Gaussian blur for smoother edges
            refined_mask = cv2.GaussianBlur(refined_mask, 
                                          (self.gaussian_blur_size, self.gaussian_blur_size), 0)
            
            # Re-threshold after blur
            _, refined_mask = cv2.threshold(refined_mask, self.final_threshold, 255, cv2.THRESH_BINARY)
        
        return refined_mask
    
    def _apply_mask_to_frame(self, frame, mask):
        """Apply binary mask to frame to extract foreground"""
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Create 3-channel mask
        mask_3ch = cv2.merge([mask, mask, mask])
        
        # Apply mask (keep foreground pixels, set background to black)
        extracted = cv2.bitwise_and(frame, mask_3ch)
        
        return extracted
    
    def _edge_aware_refinement(self, mask, frame):
        """Use edge information to refine mask boundaries"""
        # Convert frame to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges using Canny with configurable thresholds
        edges = cv2.Canny(gray, self.canny_low_threshold, self.canny_high_threshold)
        
        # Dilate edges slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        edges = cv2.dilate(edges, kernel, iterations=self.edge_dilation_iterations)
        
        # Use edges to refine mask boundaries
        # Where we have strong edges and mask boundary, trust the edge
        mask_edges = cv2.Canny(mask, self.canny_low_threshold, self.canny_high_threshold)
        
        # Combine mask with edge information
        refined = cv2.bitwise_or(mask, cv2.bitwise_and(edges, mask_edges))
        
        return refined
    
    # TODO: testing methods
    def set_parameters(self, **kwargs):
        """Set multiple parameters at once using keyword arguments"""
        valid_params = {
            # MOG2 Parameters
            'bg_history', 'bg_threshold', 'detect_shadows', 'learning_rate',
            # Motion Detection
            'motion_threshold',
            # Median Background
            'median_sample_interval', 'median_threshold',
            # Mask Combination
            'gmm_weight', 'motion_weight', 'median_weight', 'combined_mask_threshold',
            # Person Detection
            'min_contour_area', 'max_contour_area', 'min_aspect_ratio', 
            'max_aspect_ratio', 'min_solidity',
            # Morphological Operations
            'morph_kernel_size', 'small_noise_kernel_size', 'edge_dilation_size',
            # Edge and Smoothing
            'gaussian_blur_size', 'final_threshold', 'canny_low_threshold',
            'canny_high_threshold', 'edge_dilation_iterations'
        }
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
                print(f"Set {param} = {value}")
            else:
                print(f"Warning: Unknown parameter '{param}' ignored")
    
    def get_parameter_dict(self):
        """Get current parameters as a dictionary"""
        return {
            # MOG2 Parameters
            'bg_history': self.bg_history,
            'bg_threshold': self.bg_threshold,
            'detect_shadows': self.detect_shadows,
            'learning_rate': self.learning_rate,
            # Motion Detection
            'motion_threshold': self.motion_threshold,
            # Median Background
            'median_sample_interval': self.median_sample_interval,
            'median_threshold': self.median_threshold,
            # Mask Combination
            'gmm_weight': self.gmm_weight,
            'motion_weight': self.motion_weight,
            'median_weight': self.median_weight,
            'combined_mask_threshold': self.combined_mask_threshold,
            # Person Detection
            'min_contour_area': self.min_contour_area,
            'max_contour_area': self.max_contour_area,
            'min_aspect_ratio': self.min_aspect_ratio,
            'max_aspect_ratio': self.max_aspect_ratio,
            'min_solidity': self.min_solidity,
            # Morphological Operations
            'morph_kernel_size': self.morph_kernel_size,
            'small_noise_kernel_size': self.small_noise_kernel_size,
            'edge_dilation_size': self.edge_dilation_size,
            # Edge and Smoothing
            'gaussian_blur_size': self.gaussian_blur_size,
            'final_threshold': self.final_threshold,
            'canny_low_threshold': self.canny_low_threshold,
            'canny_high_threshold': self.canny_high_threshold,
            'edge_dilation_iterations': self.edge_dilation_iterations
        }
    
    def get_config_id(self):
        """Generate a unique ID based on current parameters"""
        params = self.get_parameter_dict()
        # Convert to JSON string for consistent hashing
        param_str = json.dumps(params, sort_keys=True)
        # Generate short hash
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def get_config_suffix(self, short=True):
        """Generate a descriptive suffix for filenames based on key parameters"""
        if short:
            # Short version with just key parameters
            return f"h{self.bg_history}_t{self.bg_threshold}_g{self.gmm_weight:.1f}_a{self.min_contour_area}"
        else:
            # Longer version with more parameters
            return (f"h{self.bg_history}_t{self.bg_threshold}_m{self.motion_threshold}_"
                   f"g{self.gmm_weight:.1f}_area{self.min_contour_area}_ar{self.min_aspect_ratio:.1f}")

class SimpleBackgroundSubtractor(VideoProcessor):
    """Simplified version using OpenCV's MOG2 (as allowed in FAQ)"""
    def __init__(self):
        super().__init__()
        
        # ========== ALL MOG2 CONFIGURABLE PARAMETERS ==========
        
        # Core MOG2 Parameters
        self.history = 500  # Number of frames for background model
        self.var_threshold = 16  # Threshold on squared Mahalanobis distance
        self.detect_shadows = True  # Enable shadow detection
        self.learning_rate = -1  # -1 means auto, or set 0.001-0.1
        
        # Advanced MOG2 Parameters
        self.background_ratio = 0.9  # Portion of data for background
        self.var_threshold_gen = 9  # Threshold for generating new components
        self.var_init = 15  # Initial variance of new components
        self.var_min = 4  # Minimum variance
        self.var_max = 75  # Maximum variance
        self.n_mixtures = 5  # Number of Gaussian mixtures (3-5 typically)
        self.complexity_reduction_threshold = 0.05  # Complexity reduction threshold
        
        # Shadow Detection Parameters
        self.shadow_threshold = 200  # Threshold for removing shadows (255=foreground, 127=shadow)
        
        # Post-processing Parameters
        self.morph_kernel_size = 9  # Kernel size for morphological operations
        
        # =====================================================
        
    def extract_person(self, input_path: str, extracted_path: str, binary_path: str):
        """Extract person using OpenCV's MOG2 background subtractor"""
        print("Background Subtraction: Using MOG2 (OpenCV)")
        
        # Read video
        frames, metadata = self.read_video(input_path)
        
        # Create MOG2 background subtractor with all configurable parameters
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=self.detect_shadows,
            varThreshold=self.var_threshold,
            history=self.history
        )

        # Set additional parameters using configurable values
        bg_subtractor.setBackgroundRatio(self.background_ratio)
        bg_subtractor.setVarThresholdGen(self.var_threshold_gen)
        bg_subtractor.setVarInit(self.var_init)
        bg_subtractor.setVarMin(self.var_min)
        bg_subtractor.setVarMax(self.var_max)
        bg_subtractor.setNMixtures(self.n_mixtures)
        bg_subtractor.setComplexityReductionThreshold(self.complexity_reduction_threshold)
        
        extracted_frames = []
        binary_frames = []
        
        print("Extracting foreground...")
        for i, frame in enumerate(tqdm(frames)):
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)
            
            # Remove shadows (MOG2 marks shadows as 127, foreground as 255)
            if self.detect_shadows:
                _, binary_mask = cv2.threshold(fg_mask, self.shadow_threshold, 255, cv2.THRESH_BINARY)
            else:
                binary_mask = fg_mask
            
            # Clean up the mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.morph_kernel_size, self.morph_kernel_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Fill small holes
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(binary_mask, contours, -1, 255, -1)
            
            # Create extracted frame (person only with black background)
            extracted_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
            
            extracted_frames.append(extracted_frame)
            binary_frames.append(binary_mask)
            
            if i % 30 == 0:
                print(f"Processed frame {i}/{len(frames)}")
        
        # Save results
        self.write_video(extracted_frames, extracted_path, metadata['fps'])
        self.write_video(binary_frames, binary_path, metadata['fps'])
        
        print(f"Background subtraction complete. Saved {len(frames)} frames.")