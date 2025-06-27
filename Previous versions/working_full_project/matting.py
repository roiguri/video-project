from utils import VideoProcessor
import cv2
import numpy as np
from tqdm import tqdm
import time
from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import gaussian_kde

class VideoMatter(VideoProcessor):
    def __init__(self):
        super().__init__()
        
        # Matting configuration parameters
        self.distance_exponent = 1.2
        self.separation_threshold = 10
        self.crop_vertical_radius = 600
        self.crop_horizontal_radius = 200
    
    def apply_matting(self, extracted_path: str, binary_path: str, 
                     background_path: str, matted_path: str, alpha_path: str):
        """Apply matting following the reference implementation exactly"""
        
        # Read input videos using utils
        extracted_frames, extracted_metadata = self.read_video(extracted_path)
        binary_frames, _ = self.read_video(binary_path)
        new_background = cv2.imread(background_path)
        
        if not extracted_frames or not binary_frames:
            raise ValueError("Could not read input videos")
        
        if len(extracted_frames) != len(binary_frames):
            raise ValueError("Extracted and binary videos have different frame counts")
        
        # Mathematical epsilon constant
        epsilon = 0.000000000001
        
        # Process frames
        alpha_frames = []
        matted_frames = []
        
        for current_frame_index in tqdm(range(len(extracted_frames)), desc="Video matting", leave=False, ncols=80):
            source_frame = extracted_frames[current_frame_index]
            mask_frame = binary_frames[current_frame_index]
            
            mask_grayscale = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            
            # Initialize processing masks
            foreground_region = np.zeros_like(mask_grayscale)
            background_region = np.zeros_like(mask_grayscale)
            trimap_output = np.zeros_like(mask_grayscale)
            alpha_channel = np.zeros_like(mask_grayscale)
            
            # Create initial region masks from binary input
            foreground_region[mask_grayscale >= 240] = 255
            background_region[mask_grayscale <= 10] = 255
            
            # Phase 1: Initial probability and distance computation
            bg_gradient, fg_gradient, fg_prob, bg_prob = self.compute_region_probabilities(
                source_frame, background_region, foreground_region, 
                self.crop_vertical_radius, self.crop_horizontal_radius)
            
            fg_seed_locations = np.where(foreground_region == 255)
            bg_seed_locations = np.where(background_region == 255)
            
            if len(fg_seed_locations[0]) > 0 and len(bg_seed_locations[0]) > 0:
                fg_distance_map = self.calculate_geodesic_distance(fg_gradient, fg_seed_locations)
                bg_distance_map = self.calculate_geodesic_distance(bg_gradient, bg_seed_locations)
                
                # Generate trimap from distance differences
                trimap_output[(fg_distance_map - bg_distance_map) > self.separation_threshold] = 0
                trimap_output[(bg_distance_map - fg_distance_map) > self.separation_threshold] = 255
                trimap_output[abs(bg_distance_map - fg_distance_map) <= self.separation_threshold] = 0.5 * 256
                
                # Update region masks using trimap
                background_region = np.zeros_like(mask_grayscale)
                foreground_region = np.zeros_like(mask_grayscale)
                
                foreground_region[trimap_output == 255] = 255
                background_region[trimap_output == 0] = 255
                
                # Phase 2: Refined probability and distance computation
                bg_gradient, fg_gradient, fg_prob, bg_prob = self.compute_region_probabilities(
                    source_frame, background_region, foreground_region,
                    self.crop_vertical_radius, self.crop_horizontal_radius)
                
                updated_fg_seeds = np.where(foreground_region == 255)
                updated_bg_seeds = np.where(background_region == 255)
                
                if len(updated_fg_seeds[0]) > 0 and len(updated_bg_seeds[0]) > 0:
                    fg_distance_map = self.calculate_geodesic_distance(fg_gradient, updated_fg_seeds)
                    bg_distance_map = self.calculate_geodesic_distance(bg_gradient, updated_bg_seeds)
                    
                    # Final trimap refinement
                    trimap_output[(fg_distance_map - bg_distance_map) >= self.separation_threshold] = 0
                    trimap_output[(bg_distance_map - fg_distance_map) >= self.separation_threshold] = 255
                    trimap_output[abs(bg_distance_map - fg_distance_map) < self.separation_threshold] = 0.5 * 256
                    
                    # Calculate weighted alpha values
                    foreground_weights = (fg_distance_map + epsilon) ** (-self.distance_exponent) * fg_prob
                    background_weights = (bg_distance_map + epsilon) ** (-self.distance_exponent) * bg_prob
                    
                    # Generate alpha channel
                    alpha_channel = foreground_weights / (foreground_weights + background_weights + epsilon)
                    alpha_channel[trimap_output == 255] = 1
                    alpha_channel[trimap_output == 0] = 0
                else:
                    alpha_channel = trimap_output.astype(np.float64) / 255.0
            else:
                # Fallback processing
                trimap_output = mask_grayscale
                alpha_channel = mask_grayscale.astype(np.float64) / 255.0
            
            # Composite final frame with new background in HSV space
            composite_frame = np.zeros_like(source_frame)
            composite_hsv = cv2.cvtColor(composite_frame, cv2.COLOR_BGR2HSV)
            source_hsv = cv2.cvtColor(source_frame, cv2.COLOR_BGR2HSV)
            background_hsv = cv2.cvtColor(new_background, cv2.COLOR_BGR2HSV)
            
            composite_hsv[:, :, 0] = alpha_channel * source_hsv[:, :, 0] + (1 - alpha_channel) * background_hsv[:, :, 0]
            composite_hsv[:, :, 1] = alpha_channel * source_hsv[:, :, 1] + (1 - alpha_channel) * background_hsv[:, :, 1]
            composite_hsv[:, :, 2] = alpha_channel * source_hsv[:, :, 2] + (1 - alpha_channel) * background_hsv[:, :, 2]
            
            # Convert and collect outputs
            alpha_output = (alpha_channel * 255).astype(np.uint8)
            alpha_bgr = cv2.cvtColor(alpha_output, cv2.COLOR_GRAY2BGR)
            composite_bgr = cv2.cvtColor(composite_hsv, cv2.COLOR_HSV2BGR)
            
            alpha_frames.append(np.uint8(alpha_bgr))
            matted_frames.append(np.uint8(composite_bgr))
        
        # Save videos using utils
        self.write_video(alpha_frames, alpha_path, extracted_metadata['fps'])
        self.write_video(matted_frames, matted_path, extracted_metadata['fps'])
    
    def compute_region_probabilities(self, input_frame, bg_region, fg_region, vertical_radius, horizontal_radius):
        """Calculate probability maps from reference implementation"""
        frame_hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
        fg_density_function, bg_density_function = self.estimate_color_densities(
            frame_hsv, fg_region, bg_region, vertical_radius, horizontal_radius)
        
        # Extract and clip saturation values for safe indexing
        saturation_values = frame_hsv[:, :, 1]
        clipped_saturation = np.clip(saturation_values, 0, 255).astype(int)
        
        fg_likelihood = fg_density_function[clipped_saturation]
        bg_likelihood = bg_density_function[clipped_saturation]
        
        numerical_epsilon = 0.00000000000000002
        fg_probability = fg_likelihood / (fg_likelihood + bg_likelihood + numerical_epsilon)
        bg_probability = 1 - fg_probability
        
        bg_gradient = cv2.Sobel(bg_probability, cv2.CV_64F, 1, 1, ksize=5)
        fg_gradient = cv2.Sobel(fg_probability, cv2.CV_64F, 1, 1, ksize=5)
        
        return bg_gradient, fg_gradient, fg_probability, bg_probability
    
    def estimate_color_densities(self, hsv_image, fg_region, bg_region, vertical_radius, horizontal_radius):
        """Estimate probability density functions from reference"""
        # Downsample for computational efficiency 
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
        
        # Extract region of interest around center
        window_image = downsampled_image[max(center_row - scaled_vertical, 0): min(center_row + scaled_vertical, downsampled_fg.shape[0]),
                     max(center_col - scaled_horizontal, 0): min(center_col + scaled_horizontal, downsampled_fg.shape[1])]
        window_fg_region = downsampled_fg[max(center_row - scaled_vertical, 0):min(center_row + scaled_vertical, downsampled_fg.shape[0]),
                       max(center_col - scaled_horizontal, 0):min(center_col + scaled_horizontal, downsampled_fg.shape[1])]
        window_bg_region = downsampled_bg[max(center_row - scaled_vertical, 0):min(center_row + scaled_vertical, downsampled_fg.shape[0]),
                       max(center_col - scaled_horizontal, 0):min(center_col + scaled_horizontal, downsampled_fg.shape[1])]
        
        # Generate evaluation grid for KDE
        intensity_range = np.linspace(0, 255, 256)
        hue_channel, saturation_channel, value_channel = cv2.split(window_image)
        
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
    
    def calculate_geodesic_distance(self, gradient_image, seed_locations):
        """Calculate geodesic distance from reference implementation"""
        if len(seed_locations) == 0 or len(seed_locations[0]) == 0:
            return np.ones(gradient_image.shape) * np.inf
            
        seed_row, seed_col = seed_locations[0], seed_locations[1]
        invalid_regions = self.detect_missing_mask(gradient_image)
        
        # Use Euclidean distance transform if no invalid regions
        if invalid_regions.sum() == 0:
            distance_array = np.ones(gradient_image.shape)
            distance_array[seed_row, seed_col] = 0
            return distance_transform_edt(distance_array)
        
        valid_pixel_count = (1 - invalid_regions).sum()
        distance_map = np.ones(gradient_image.shape) * np.inf
        distance_map[seed_row, seed_col] = 0
        
        def shift_image_direction(image_array, direction_name):
            if direction_name == 'n':
                boundary_row = image_array[0, :].copy()
                image_array = np.roll(image_array, 1, axis=0)
                image_array[0, :] = boundary_row
            elif direction_name == 's':
                boundary_row = image_array[-1, :].copy()
                image_array = np.roll(image_array, -1, axis=0)
                image_array[-1, :] = boundary_row
            elif direction_name == 'e':
                boundary_col = image_array[:, 0].copy()
                image_array = np.roll(image_array, 1, axis=1)
                image_array[:, 0] = boundary_col
            elif direction_name == 'w':
                boundary_col = image_array[:, -1].copy()
                image_array = np.roll(image_array, -1, axis=1)
                image_array[:, -1] = boundary_col
            elif direction_name == 'ne':
                image_array = shift_image_direction(image_array, 'n')
                image_array = shift_image_direction(image_array, 'e')
            elif direction_name == 'nw':
                image_array = shift_image_direction(image_array, 'n')
                image_array = shift_image_direction(image_array, 'w')
            elif direction_name == 'sw':
                image_array = shift_image_direction(image_array, 's')
                image_array = shift_image_direction(image_array, 'w')
            elif direction_name == 'se':
                image_array = shift_image_direction(image_array, 's')
                image_array = shift_image_direction(image_array, 'e')
            return image_array
        
        def perform_expansion_iteration(distance_array):
            diagonal_cost = np.sqrt(2)
            shifted_results = []
            movement_directions = ['n', 's', 'e', 'w', 'ne', 'nw', 'sw', 'se']
            movement_costs = [1, 1, 1, 1, diagonal_cost, diagonal_cost, diagonal_cost, diagonal_cost]
            
            for direction, cost in zip(movement_directions, movement_costs):
                shifted_distance = shift_image_direction(distance_array.copy(), direction) + cost
                shifted_distance = np.minimum(shifted_distance, distance_array)
                shifted_results.append(shifted_distance)
            
            # Combine results by taking minimum
            combined_result = shifted_results[0]
            for result in shifted_results[1:]:
                combined_result = np.minimum(combined_result, result)
            return combined_result
        
        # Iterative distance propagation
        previous_distance = distance_map.copy()
        iteration_limit = 1000
        for _ in range(iteration_limit):
            expanded_distance = perform_expansion_iteration(distance_map)
            distance_map = np.where(invalid_regions, distance_map, expanded_distance)
            processed_pixels = distance_map.size - len(np.where(distance_map == np.inf)[0])
            
            if processed_pixels >= valid_pixel_count or np.allclose(previous_distance, distance_map):
                break
            previous_distance = distance_map.copy()
        
        return distance_map
    
    def calculate_kde(self, data_values, evaluation_grid, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scipy"""
        if len(data_values) == 0:
            return np.ones_like(evaluation_grid)
        density_estimator = gaussian_kde(data_values, bw_method=bandwidth / (data_values.std(ddof=1) + 1e-10), **kwargs)
        return density_estimator.evaluate(evaluation_grid)
    
    def detect_missing_mask(self, data_slab):
        """Get missing mask from array"""
        nan_locations = np.where(np.isnan(data_slab), 1, 0)
        if not hasattr(data_slab, 'mask'):
            masked_locations = np.zeros(data_slab.shape)
        else:
            if data_slab.mask.size == 1 and data_slab.mask == False:
                masked_locations = np.zeros(data_slab.shape)
            else:
                masked_locations = np.where(data_slab.mask, 1, 0)
        combined_mask = np.where(masked_locations + nan_locations > 0, 1, 0)
        return combined_mask