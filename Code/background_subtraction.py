from utils import VideoUtils
import cv2
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde

# Background subtraction parameters
BG_SAMPLE_COUNT = 20
FG_SAMPLE_COUNT = 30
ITERATION_LIMIT = 5
KDE_BANDWIDTH = 0.95

class BackgroundSubtractor(VideoUtils):
    def __init__(self):
        super().__init__()
    
    def apply_subtraction(self, video_path, bg_img_path, extracted_path, binary_path, utils):
        """Main background subtraction function"""
        
        np.random.seed(0)
        
        # Read video frames
        frames, metadata = self.read_video(video_path)
        
        if not frames:
            raise ValueError(f"No frames found in {video_path}")
        
        # Stage 1: Create initial masks using KNN background subtractor
        initial_masks = self.stage1(frames)
        
        # Stage 2: Improve masks with morphology and collect color samples
        improved_masks, bg_samples, fg_samples = self.stage2(frames, initial_masks)
        
        # Stage 3: Generate final frames using KDE probability classification
        extracted_frames, binary_frames = self.stage3(
            frames, improved_masks, bg_samples, fg_samples)
        
        # Save output videos
        self.write_video(extracted_frames, extracted_path, metadata['fps'])
        self.write_video(binary_frames, binary_path, metadata['fps'])
        utils.record_timing('time_to_binary')
        
        return binary_frames, extracted_frames
    
    def stage1(self, frames):
        """Stage 1: Initial background subtraction using KNN"""
        knn = cv2.createBackgroundSubtractorKNN()
        masks = []
        
        # Run multiple iterations to stabilize background model
        for iter_num in range(ITERATION_LIMIT):
            iter_masks = []
            
            for frame in tqdm(frames, desc=f"KNN iteration {iter_num + 1}/{ITERATION_LIMIT}", leave=False, ncols=80):
                # Convert to HSV and use only saturation and value channels
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                sv = hsv[:, :, 1:]  # Skip hue channel for better stability
                
                # Apply KNN background subtractor
                fg_mask = knn.apply(sv)
                fg_mask = (fg_mask > 130).astype(np.uint8)  # Threshold foreground mask
                iter_masks.append(fg_mask)
            
            masks = iter_masks  # Keep masks from latest iteration
        
        return masks
    
    def stage2(self, frames, masks):
        """Stage 2: Improve masks using morphological operations and collect color samples"""
        num_frames = len(frames)
        bg_samples = np.empty((BG_SAMPLE_COUNT * num_frames, 3))
        fg_samples = np.empty((FG_SAMPLE_COUNT * num_frames, 3))
        
        fg_idx = 0
        bg_idx = 0
        improved_masks = []
        
        for i, frame in enumerate(tqdm(frames, desc="Improving masks", leave=False, ncols=80)):
            mask = masks[i]
            
            # Apply morphological closing to fill holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours and keep only the largest connected component
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                clean_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.fillPoly(clean_mask, pts=[largest_contour], color=1)
            else:
                clean_mask = np.zeros(mask.shape, dtype=np.uint8)
            
            improved_masks.append(clean_mask)
            
            # Collect color samples from foreground and background regions
            fg_locations, fg_count = self.sample_pixel_locations(clean_mask, 1, FG_SAMPLE_COUNT)
            bg_locations, bg_count = self.sample_pixel_locations(clean_mask, 0, BG_SAMPLE_COUNT)
            
            # Store foreground color samples
            if fg_count > 0:
                fg_samples[fg_idx:fg_idx + fg_count] = frame[fg_locations[:, 0], fg_locations[:, 1], :]
                fg_idx += fg_count
            
            # Store background color samples
            if bg_count > 0:
                bg_samples[bg_idx:bg_idx + bg_count] = frame[bg_locations[:, 0], bg_locations[:, 1], :]
                bg_idx += bg_count
        
        # Trim arrays to actual sample count
        bg_samples = bg_samples[:bg_idx]
        fg_samples = fg_samples[:fg_idx]
        
        return improved_masks, bg_samples, fg_samples
    
    def stage3(self, frames, masks, bg_samples, fg_samples):
        """Stage 3: Generate final frames using KDE probability classification"""
        # Build KDE models from collected color samples
        fg_kde = gaussian_kde(np.asarray(fg_samples).T, bw_method=KDE_BANDWIDTH)
        bg_kde = gaussian_kde(np.asarray(bg_samples).T, bw_method=KDE_BANDWIDTH)
        
        # Memoization dictionaries for probability caching
        fg_cache = dict()
        bg_cache = dict()
        
        binary_results = []
        extracted_results = []
        
        for i, frame in enumerate(tqdm(frames, desc="Generating frames with KDE", leave=False, ncols=80)):
            mask = masks[i].copy()
            final_mask = np.zeros_like(mask)
            pixel_positions = np.where(mask == 1)
            
            # Extract pixel colors from candidate foreground regions
            colors = frame[pixel_positions]
            
            # Calculate foreground and background probabilities for each pixel
            fg_probabilities = np.array([
                self.check_probability(fg_cache, tuple(color), fg_kde)
                for color in colors
            ])
            bg_probabilities = np.array([
                self.check_probability(bg_cache, tuple(color), bg_kde)
                for color in colors
            ])
            # Classify pixels based on probability comparison
            final_mask[pixel_positions] = (fg_probabilities > bg_probabilities).astype(np.uint8)
            
            # Apply erosion to remove noise
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            final_mask = cv2.erode(final_mask, erode_kernel).astype(np.uint8)
            
            # Keep only the largest connected component
            contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                result_mask = np.zeros((final_mask.shape), dtype=np.uint8)
                cv2.fillPoly(result_mask, pts=[largest_contour], color=1)
                # Apply closing to fill remaining holes
                result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, np.ones((15, 15))).astype(np.uint8)
            else:
                result_mask = final_mask
            
            # Convert to 8-bit mask and extract foreground
            result_mask[result_mask == 1] = 255
            binary_results.append(result_mask)
            extracted_results.append(cv2.bitwise_and(frame, frame, mask=result_mask))
        
        return extracted_results, binary_results
    
    def sample_pixel_locations(self, mask, target_value, count):
        """Sample random pixel locations from mask regions"""
        found_indices = np.where(mask == target_value)
        if len(found_indices[0]) < count:
            print(f"Not enough points in mask, using {len(found_indices[0])} points instead of {count}")
            count = len(found_indices[0])
        if count > 0:
            # Randomly select pixel locations from available candidates
            selection = np.random.choice(len(found_indices[0]), count)
            return np.column_stack((found_indices[0][selection], found_indices[1][selection])), count
        else:
            return np.array([]).reshape(0, 2), 0
    
    def check_probability(self, cache, color, kde_func):
        """Check probability with caching"""
        if color in cache:
            return cache[color]
        else:
            cache[color] = kde_func(color)[0]
            return cache[color]