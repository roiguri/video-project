from utils import VideoUtils
import cv2
import numpy as np
from tqdm import tqdm
import time
from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import gaussian_kde

# Matting parameters
DISTANCE_EXPONENT = 1.2
SEPARATION_THRESHOLD = 10
CROP_VERTICAL_RADIUS = 600
CROP_HORIZONTAL_RADIUS = 200
KDE_BANDWIDTH = 0.2
EPSILON = 1e-12


class VideoMatter(VideoUtils):
    def __init__(self):
        super().__init__()
    
    def apply_matting(self, extracted_path: str, binary_path: str, 
                     background_path: str, matted_path: str, alpha_path: str, utils):
        """Apply matting to video"""
        
        # Read input videos
        extracted_frames, extracted_metadata = self.read_video(extracted_path)
        binary_frames, _ = self.read_video(binary_path)
        new_background = cv2.imread(background_path)
        
        if not extracted_frames or not binary_frames:
            raise ValueError("Could not read input videos")
        
        if len(extracted_frames) != len(binary_frames):
            raise ValueError("Extracted and binary videos have different frame counts")
        
        
        alpha_frames = []
        matted_frames = []
        
        for current_frame_index in tqdm(range(len(extracted_frames)), desc="Video matting", leave=False, ncols=80):
            source_frame = extracted_frames[current_frame_index]
            mask_frame = binary_frames[current_frame_index]
            
            mask_grayscale = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            image_shape = mask_grayscale.shape
            
            # Initialize processing masks
            foreground_region = np.zeros_like(mask_grayscale)
            background_region = np.zeros_like(mask_grayscale)
            trimap_output = np.zeros_like(mask_grayscale)
            alpha_channel = np.zeros_like(mask_grayscale)
            
            # Create initial region masks
            foreground_region[mask_grayscale >= 240] = 255
            background_region[mask_grayscale <= 10] = 255
            
            # Phase 1: Initial computation
            fg_prob, bg_prob = self.compute_region_probabilities(
                source_frame, background_region, foreground_region,
                CROP_VERTICAL_RADIUS, CROP_HORIZONTAL_RADIUS)
            
            fg_seed_locations = np.where(foreground_region == 255)
            bg_seed_locations = np.where(background_region == 255)
            
            if len(fg_seed_locations[0]) > 0 and len(bg_seed_locations[0]) > 0:
                fg_distance_map = self.calculate_geodesic_distance(fg_seed_locations, image_shape)
                bg_distance_map = self.calculate_geodesic_distance(bg_seed_locations, image_shape)
                
                # Generate trimap
                trimap_output[(fg_distance_map - bg_distance_map) > SEPARATION_THRESHOLD] = 0
                trimap_output[(bg_distance_map - fg_distance_map) > SEPARATION_THRESHOLD] = 255
                trimap_output[abs(bg_distance_map - fg_distance_map) <= SEPARATION_THRESHOLD] = 128
                
                # Update region masks
                background_region = np.zeros_like(mask_grayscale)
                foreground_region = np.zeros_like(mask_grayscale)
                
                foreground_region[trimap_output == 255] = 255
                background_region[trimap_output == 0] = 255
                
                # Phase 2: Refined computation
                fg_prob, bg_prob = self.compute_region_probabilities(
                    source_frame, background_region, foreground_region,
                    CROP_VERTICAL_RADIUS, CROP_HORIZONTAL_RADIUS)
                
                updated_fg_seeds = np.where(foreground_region == 255)
                updated_bg_seeds = np.where(background_region == 255)
                
                if len(updated_fg_seeds[0]) > 0 and len(updated_bg_seeds[0]) > 0:
                    fg_distance_map = self.calculate_geodesic_distance(updated_fg_seeds, image_shape)
                    bg_distance_map = self.calculate_geodesic_distance(updated_bg_seeds, image_shape)
                    
                    # Final trimap
                    trimap_output[(fg_distance_map - bg_distance_map) >= SEPARATION_THRESHOLD] = 0
                    trimap_output[(bg_distance_map - fg_distance_map) >= SEPARATION_THRESHOLD] = 255
                    trimap_output[abs(bg_distance_map - fg_distance_map) < SEPARATION_THRESHOLD] = 0.5 * 256
                    
                    # Calculate alpha weights
                    foreground_weights = (fg_distance_map + EPSILON) ** (-DISTANCE_EXPONENT) * fg_prob
                    background_weights = (bg_distance_map + EPSILON) ** (-DISTANCE_EXPONENT) * bg_prob
                    
                    # Generate alpha
                    alpha_channel = foreground_weights / (foreground_weights + background_weights + EPSILON)
                    alpha_channel[trimap_output == 255] = 1
                    alpha_channel[trimap_output == 0] = 0
                else:
                    alpha_channel = trimap_output.astype(np.float64) / 255.0
            else:
                trimap_output = mask_grayscale
                alpha_channel = mask_grayscale.astype(np.float64) / 255.0
            
            # Composite frame in HSV space
            composite_frame = np.zeros_like(source_frame)
            composite_hsv = cv2.cvtColor(composite_frame, cv2.COLOR_BGR2HSV)
            source_hsv = cv2.cvtColor(source_frame, cv2.COLOR_BGR2HSV)
            background_hsv = cv2.cvtColor(new_background, cv2.COLOR_BGR2HSV)
            
            composite_hsv[:, :, 0] = alpha_channel * source_hsv[:, :, 0] + (1 - alpha_channel) * background_hsv[:, :, 0]
            composite_hsv[:, :, 1] = alpha_channel * source_hsv[:, :, 1] + (1 - alpha_channel) * background_hsv[:, :, 1]
            composite_hsv[:, :, 2] = alpha_channel * source_hsv[:, :, 2] + (1 - alpha_channel) * background_hsv[:, :, 2]
            
            # Convert outputs
            alpha_output = (alpha_channel * 255).astype(np.uint8)
            alpha_bgr = cv2.cvtColor(alpha_output, cv2.COLOR_GRAY2BGR)
            composite_bgr = cv2.cvtColor(composite_hsv, cv2.COLOR_HSV2BGR)
            
            alpha_frames.append(np.uint8(alpha_bgr))
            matted_frames.append(np.uint8(composite_bgr))
        
        # Save videos
        self.write_video(alpha_frames, alpha_path, extracted_metadata['fps'])
        utils.record_timing('time_to_alpha')
        self.write_video(matted_frames, matted_path, extracted_metadata['fps'])
        utils.record_timing('time_to_matted')
    
    def compute_region_probabilities(self, input_frame, bg_region, fg_region, vertical_radius, horizontal_radius):
        """Calculate probability maps"""
        frame_hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
        fg_density_function, bg_density_function = self.estimate_color_densities(
            frame_hsv, fg_region, bg_region, vertical_radius, horizontal_radius)
        
        # Extract and clip saturation values
        saturation_values = frame_hsv[:, :, 1]
        clipped_saturation = np.clip(saturation_values, 0, 255).astype(int)
        
        fg_likelihood = fg_density_function[clipped_saturation]
        bg_likelihood = bg_density_function[clipped_saturation]
        
        fg_probability = fg_likelihood / (fg_likelihood + bg_likelihood + EPSILON)
        bg_probability = 1 - fg_probability
        
        return fg_probability, bg_probability
    
    def estimate_color_densities(self, hsv_image, fg_region, bg_region, vertical_radius, horizontal_radius):
        """Estimate color density functions"""
        # Downsample for efficiency
        downsampled_image = cv2.resize(hsv_image, (hsv_image.shape[1] // 4, hsv_image.shape[0] // 4))
        downsampled_fg = cv2.resize(fg_region, (fg_region.shape[1] // 4, fg_region.shape[0] // 4))
        downsampled_bg = cv2.resize(bg_region, (bg_region.shape[1] // 4, bg_region.shape[0] // 4))
        scaled_vertical = vertical_radius // 4
        scaled_horizontal = horizontal_radius // 4
        
        # Locate object center
        fg_row_indices, fg_col_indices = np.where(downsampled_fg == 255)
        if len(fg_row_indices) == 0:
            return np.ones(256), np.ones(256)
            
        center_row = int(np.mean(fg_row_indices))
        center_col = int(np.mean(fg_col_indices))
        
        # Extract region of interest
        window_image = downsampled_image[max(center_row - scaled_vertical, 0): min(center_row + scaled_vertical, downsampled_fg.shape[0]),
                     max(center_col - scaled_horizontal, 0): min(center_col + scaled_horizontal, downsampled_fg.shape[1])]
        window_fg_region = downsampled_fg[max(center_row - scaled_vertical, 0):min(center_row + scaled_vertical, downsampled_fg.shape[0]),
                       max(center_col - scaled_horizontal, 0):min(center_col + scaled_horizontal, downsampled_fg.shape[1])]
        window_bg_region = downsampled_bg[max(center_row - scaled_vertical, 0):min(center_row + scaled_vertical, downsampled_fg.shape[0]),
                       max(center_col - scaled_horizontal, 0):min(center_col + scaled_horizontal, downsampled_fg.shape[1])]
        
        # Generate evaluation grid
        intensity_range = np.linspace(0, 255, 256)
        _, saturation_channel, _ = cv2.split(window_image)
        
        # Calculate foreground density
        fg_pixel_rows, fg_pixel_cols = np.where(window_fg_region == 255)
        if len(fg_pixel_rows) > 0:
            fg_saturation_data = saturation_channel[fg_pixel_rows, fg_pixel_cols]
            try:
                fg_density = self.calculate_kde(fg_saturation_data, intensity_range)
            except:
                fg_density = np.ones(256)
        else:
            fg_density = np.ones(256)
            
        # Calculate background density
        bg_pixel_rows, bg_pixel_cols = np.where(window_bg_region == 255)
        if len(bg_pixel_rows) > 0:
            bg_saturation_data = saturation_channel[bg_pixel_rows, bg_pixel_cols]
            try:
                bg_density = self.calculate_kde(bg_saturation_data, intensity_range)
            except:
                bg_density = np.ones(256)
        else:
            bg_density = np.ones(256)
            
        return fg_density, bg_density
    
    def calculate_geodesic_distance(self, seed_locations, image_shape):
        """Calculate distance from seed locations"""
        if len(seed_locations) == 0 or len(seed_locations[0]) == 0:
            return np.ones(image_shape) * np.inf
            
        seed_row, seed_col = seed_locations[0], seed_locations[1]
        
        # Euclidean distance transform
        distance_array = np.ones(image_shape)
        distance_array[seed_row, seed_col] = 0
        return distance_transform_edt(distance_array)
    
    def calculate_kde(self, data_values, evaluation_grid, **kwargs):
        """Kernel density estimation"""
        if len(data_values) == 0:
            return np.ones_like(evaluation_grid)
        # TODO: kwargs?
        density_estimator = gaussian_kde(data_values, bw_method=KDE_BANDWIDTH / (data_values.std(ddof=1) + 1e-10), **kwargs)
        return density_estimator.evaluate(evaluation_grid)