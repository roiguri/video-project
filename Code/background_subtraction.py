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
    
    def apply_subtraction(self, input_video_path, background_img_path, extracted_output_path, binary_output_path, utils):
        """Main background subtraction function"""
        
        np.random.seed(0)
        
        # Read video frames
        video_frames, metadata = self.read_video(input_video_path)
        
        if not video_frames:
            raise ValueError(f"No frames found in {input_video_path}")
        
        # Stage 1: Create initial masks
        preliminary_masks = self.stage1(video_frames)
        
        # Stage 2: Improve masks
        refined_masks, bg_samples, fg_samples = self.stage2(video_frames, preliminary_masks)
        
        # Stage 3: Generate final frames
        extracted_sequence, binary_sequence = self.stage3(
            video_frames, refined_masks, bg_samples, fg_samples)
        
        # Save output videos
        self.write_video(extracted_sequence, extracted_output_path, metadata['fps'])
        self.write_video(binary_sequence, binary_output_path, metadata['fps'])
        utils.record_timing('time_to_binary')
        
        return binary_sequence, extracted_sequence
    
    def stage1(self, video_frames):
        knn_model = cv2.createBackgroundSubtractorKNN()
        mask_sequence = []
        
        for iteration_num in range(ITERATION_LIMIT):
            current_iteration_masks = []
            
            for current_frame in tqdm(video_frames, desc=f"KNN iteration {iteration_num + 1}/{ITERATION_LIMIT}", leave=False, ncols=80):
                # Convert to HSV and use S and V channels
                hsv_converted = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
                sv_channels = hsv_converted[:, :, 1:]
                
                # Apply background subtractor
                foreground_mask = knn_model.apply(sv_channels)
                foreground_mask = (foreground_mask > 130).astype(np.uint8)
                current_iteration_masks.append(foreground_mask)
            
            mask_sequence = current_iteration_masks
        
        return mask_sequence
    
    def stage2(self, video_frames, mask_sequence):
        """Improve masks using morphological operations"""
        total_frames = len(video_frames)
        bg_samples = np.empty((BG_SAMPLE_COUNT * total_frames, 3))
        fg_samples = np.empty((FG_SAMPLE_COUNT * total_frames, 3))
        
        fg_index = 0
        bg_index = 0
        refined_masks = []
        
        for frame_idx, current_frame in enumerate(tqdm(video_frames, desc="Improving masks", leave=False, ncols=80)):
            current_mask = mask_sequence[frame_idx]
            
            # Morphological operations
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, morph_kernel)
            
            # Find contours and get largest
            detected_contours, _ = cv2.findContours(current_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if detected_contours:
                sorted_contours = sorted(detected_contours, key=cv2.contourArea, reverse=True)
                object_mask = np.zeros(current_mask.shape, dtype=np.uint8)
                cv2.fillPoly(object_mask, pts=[sorted_contours[0]], color=1)
            else:
                object_mask = np.zeros(current_mask.shape, dtype=np.uint8)
            
            refined_masks.append(object_mask)
            
            # Collect samples
            fg_locations, actual_fg_count = self.sample_pixel_locations(object_mask, 1, FG_SAMPLE_COUNT)
            bg_locations, actual_bg_count = self.sample_pixel_locations(object_mask, 0, BG_SAMPLE_COUNT)
            
            if actual_fg_count > 0:
                fg_samples[fg_index:fg_index + actual_fg_count] = current_frame[fg_locations[:, 0], fg_locations[:, 1], :]
                fg_index += actual_fg_count
            
            if actual_bg_count > 0:
                bg_samples[bg_index:bg_index + actual_bg_count] = current_frame[bg_locations[:, 0], bg_locations[:, 1], :]
                bg_index += actual_bg_count
        
        bg_samples = bg_samples[:bg_index]
        fg_samples = fg_samples[:fg_index]
        
        return refined_masks, bg_samples, fg_samples
    
    def stage3(self, video_frames, mask_sequence, bg_samples, fg_samples):
        """Generate final frames using KDE"""
        fg_probability_model = gaussian_kde(np.asarray(fg_samples).T, bw_method=KDE_BANDWIDTH)
        bg_probability_model = gaussian_kde(np.asarray(bg_samples).T, bw_method=KDE_BANDWIDTH)
        
        # Memoization dictionaries
        fg_prob_cache = dict()
        bg_prob_cache = dict()
        
        binary_sequence = []
        extracted_sequence = []
        
        for frame_idx, current_frame in enumerate(tqdm(video_frames, desc="Generating frames with KDE", leave=False, ncols=80)):
            current_mask = mask_sequence[frame_idx].copy()
            refined_mask = np.zeros_like(current_mask)
            pixel_positions = np.where(current_mask == 1)
            
            # Extract pixel colors
            pixel_colors = current_frame[pixel_positions]
            
            # Check probability of each pixel
            fg_probabilities = np.array([
                self.check_probability(fg_prob_cache, tuple(color), fg_probability_model)
                for color in pixel_colors
            ])
            bg_probabilities = np.array([
                self.check_probability(bg_prob_cache, tuple(color), bg_probability_model)
                for color in pixel_colors
            ])
            refined_mask[pixel_positions] = (fg_probabilities > bg_probabilities).astype(np.uint8)
            
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_mask = cv2.erode(refined_mask, erosion_kernel).astype(np.uint8)
            
            final_contours, _ = cv2.findContours(refined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if final_contours:
                largest_contours = sorted(final_contours, key=cv2.contourArea, reverse=True)
                object_mask = np.zeros((refined_mask.shape), dtype=np.uint8)
                cv2.fillPoly(object_mask, pts=[largest_contours[0]], color=1)
                object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((15, 15))).astype(np.uint8)
            else:
                object_mask = refined_mask
            
            object_mask[object_mask == 1] = 255
            binary_sequence.append(object_mask)
            extracted_sequence.append(cv2.bitwise_and(current_frame, current_frame, mask=object_mask))
        
        return extracted_sequence, binary_sequence
    
    def sample_pixel_locations(self, source_mask, target_value, desired_count):
        """Sample pixel locations from mask"""
        found_indices = np.where(source_mask == target_value)
        if len(found_indices[0]) < desired_count:
            print(f"Not enough points in source_mask, using {len(found_indices[0])} points instead of {desired_count}")
            desired_count = len(found_indices[0])
        if desired_count > 0:
            random_selection = np.random.choice(len(found_indices[0]), desired_count)
            return np.column_stack((found_indices[0][random_selection], found_indices[1][random_selection])), desired_count
        else:
            return np.array([]).reshape(0, 2), 0
    
    def check_probability(self, probability_cache, color_value, probability_function):
        """Check probability with caching"""
        if color_value in probability_cache:
            return probability_cache[color_value]
        else:
            probability_cache[color_value] = probability_function(color_value)[0]
            return probability_cache[color_value]