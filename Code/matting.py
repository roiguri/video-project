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
        frames, metadata = self.read_video(extracted_path)
        masks, _ = self.read_video(binary_path)
        bg_image = cv2.imread(background_path)
        
        if not frames or not masks:
            raise ValueError("Could not read input videos")
        
        if len(frames) != len(masks):
            raise ValueError("Extracted and binary videos have different frame counts")
        
        
        alpha_results = []
        matted_results = []
        
        for i in tqdm(range(len(frames)), desc="Video matting", leave=False, ncols=80):
            frame = frames[i]
            mask = masks[i]
            
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            h, w = gray_mask.shape
            
            # Initialize processing masks
            foreground_region = np.zeros_like(gray_mask)
            background_region = np.zeros_like(gray_mask)
            trimap = np.zeros_like(gray_mask)
            alpha = np.zeros_like(gray_mask)
            
            # Create initial region masks from binary mask
            foreground_region[gray_mask >= 240] = 255  # High confidence foreground
            background_region[gray_mask <= 10] = 255   # High confidence background
            
            # Phase 1: Initial computation using rough regions
            fg_prob, bg_prob = self.compute_region_probabilities(
                frame, background_region, foreground_region,
                CROP_VERTICAL_RADIUS, CROP_HORIZONTAL_RADIUS)
            
            fg_seed_locations = np.where(foreground_region == 255)
            bg_seed_locations = np.where(background_region == 255)
            
            if len(fg_seed_locations[0]) > 0 and len(bg_seed_locations[0]) > 0:
                # Calculate distance maps from foreground and background seed points
                fg_distance_map = self.calculate_distance(fg_seed_locations, (h, w))
                bg_distance_map = self.calculate_distance(bg_seed_locations, (h, w))
                
                # Generate trimap based on distance comparison
                trimap[(fg_distance_map - bg_distance_map) > SEPARATION_THRESHOLD] = 0    # Background
                trimap[(bg_distance_map - fg_distance_map) > SEPARATION_THRESHOLD] = 255  # Foreground
                trimap[abs(bg_distance_map - fg_distance_map) <= SEPARATION_THRESHOLD] = 128  # Unknown
                
                # Update region masks based on trimap
                background_region = np.zeros_like(gray_mask)
                foreground_region = np.zeros_like(gray_mask)
                
                foreground_region[trimap == 255] = 255
                background_region[trimap == 0] = 255
                
                # Phase 2: Refined computation with better region definitions
                fg_prob, bg_prob = self.compute_region_probabilities(
                    frame, background_region, foreground_region,
                    CROP_VERTICAL_RADIUS, CROP_HORIZONTAL_RADIUS)
                
                new_fg_seed_locations = np.where(foreground_region == 255)
                new_bg_seed_locations = np.where(background_region == 255)
                
                if len(new_fg_seed_locations[0]) > 0 and len(new_bg_seed_locations[0]) > 0:
                    # Recalculate distance maps with refined seed points
                    fg_distance_map = self.calculate_distance(new_fg_seed_locations, (h, w))
                    bg_distance_map = self.calculate_distance(new_bg_seed_locations, (h, w))
                    
                    # Final trimap with refined boundaries
                    trimap[(fg_distance_map - bg_distance_map) >= SEPARATION_THRESHOLD] = 0
                    trimap[(bg_distance_map - fg_distance_map) >= SEPARATION_THRESHOLD] = 255
                    trimap[abs(bg_distance_map - fg_distance_map) < SEPARATION_THRESHOLD] = 0.5 * 256
                    
                    # Calculate alpha weights combining distance and color probability
                    fg_weights = (fg_distance_map + EPSILON) ** (-DISTANCE_EXPONENT) * fg_prob
                    bg_weights = (bg_distance_map + EPSILON) ** (-DISTANCE_EXPONENT) * bg_prob
                    
                    # Generate alpha matte using weighted combination
                    alpha = fg_weights / (fg_weights + bg_weights + EPSILON)
                    alpha[trimap == 255] = 1  # Definite foreground
                    alpha[trimap == 0] = 0    # Definite background
                else:
                    alpha = trimap.astype(np.float64) / 255.0
            else:
                trimap = gray_mask
                alpha = gray_mask.astype(np.float64) / 255.0
            
            # Composite frame in HSV space for better color blending
            result = np.zeros_like(frame)
            result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bg_hsv = cv2.cvtColor(bg_image, cv2.COLOR_BGR2HSV)
            
            # Alpha blend each HSV channel separately
            result_hsv[:, :, 0] = alpha * frame_hsv[:, :, 0] + (1 - alpha) * bg_hsv[:, :, 0]  # Hue
            result_hsv[:, :, 1] = alpha * frame_hsv[:, :, 1] + (1 - alpha) * bg_hsv[:, :, 1]  # Saturation
            result_hsv[:, :, 2] = alpha * frame_hsv[:, :, 2] + (1 - alpha) * bg_hsv[:, :, 2]  # Value
            
            # Convert outputs to BGR for saving
            alpha_img = (alpha * 255).astype(np.uint8)
            alpha_bgr = cv2.cvtColor(alpha_img, cv2.COLOR_GRAY2BGR)
            result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
            
            alpha_results.append(np.uint8(alpha_bgr))
            matted_results.append(np.uint8(result_bgr))
        
        # Save videos
        self.write_video(alpha_results, alpha_path, metadata['fps'])
        utils.record_timing('time_to_alpha')
        self.write_video(matted_results, matted_path, metadata['fps'])
        utils.record_timing('time_to_matted')
    
    def compute_region_probabilities(self, frame, background_region, foreground_region, v_radius, h_radius):
        """Calculate probability maps"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fg_density, bg_density = self.estimate_color_densities(
            hsv, foreground_region, background_region, v_radius, h_radius)
        
        # Extract and clip saturation values
        sat = hsv[:, :, 1]
        sat_clipped = np.clip(sat, 0, 255).astype(int)
        
        fg_like = fg_density[sat_clipped]
        bg_like = bg_density[sat_clipped]
        
        fg_prob = fg_like / (fg_like + bg_like + EPSILON)
        bg_prob = 1 - fg_prob
        
        return fg_prob, bg_prob
    
    def estimate_color_densities(self, hsv, foreground_region, background_region, v_radius, h_radius):
        """Estimate color density functions using KDE on saturation channel"""
        # Downsample for computational efficiency
        small_hsv = cv2.resize(hsv, (hsv.shape[1] // 4, hsv.shape[0] // 4))
        small_fg = cv2.resize(foreground_region, (foreground_region.shape[1] // 4, foreground_region.shape[0] // 4))
        small_bg = cv2.resize(background_region, (background_region.shape[1] // 4, background_region.shape[0] // 4))
        v_small = v_radius // 4
        h_small = h_radius // 4
        
        # Locate object center from foreground region
        rows, cols = np.where(small_fg == 255)
        if len(rows) == 0:
            return np.ones(256), np.ones(256)
            
        center_y = int(np.mean(rows))
        center_x = int(np.mean(cols))
        
        # Extract focused region of interest around object center
        y1, y2 = max(center_y - v_small, 0), min(center_y + v_small, small_fg.shape[0])
        x1, x2 = max(center_x - h_small, 0), min(center_x + h_small, small_fg.shape[1])
        
        window_hsv = small_hsv[y1:y2, x1:x2]
        window_fg = small_fg[y1:y2, x1:x2]
        window_bg = small_bg[y1:y2, x1:x2]
        
        # Generate evaluation grid for saturation values (0-255)
        eval_range = np.linspace(0, 255, 256)
        _, sat_channel, _ = cv2.split(window_hsv)
        
        # Calculate foreground color density from saturation values
        fg_y, fg_x = np.where(window_fg == 255)
        if len(fg_y) > 0:
            fg_sat_data = sat_channel[fg_y, fg_x]
            try:
                fg_density = self.calculate_kde(fg_sat_data, eval_range)
            except:
                fg_density = np.ones(256)  # Fallback to uniform distribution
        else:
            fg_density = np.ones(256)
            
        # Calculate background color density from saturation values
        bg_y, bg_x = np.where(window_bg == 255)
        if len(bg_y) > 0:
            bg_sat_data = sat_channel[bg_y, bg_x]
            try:
                bg_density = self.calculate_kde(bg_sat_data, eval_range)
            except:
                bg_density = np.ones(256)  # Fallback to uniform distribution
        else:
            bg_density = np.ones(256)
            
        return fg_density, bg_density
    
    def calculate_distance(self, seeds, shape):
        """Calculate distance from seed locations"""
        if len(seeds) == 0 or len(seeds[0]) == 0:
            return np.ones(shape) * np.inf
            
        seed_y, seed_x = seeds[0], seeds[1]
        
        # Euclidean distance transform
        dist_map = np.ones(shape)
        dist_map[seed_y, seed_x] = 0
        return distance_transform_edt(dist_map)
    
    def calculate_kde(self, data, eval_grid):
        """Kernel density estimation"""
        if len(data) == 0:
            return np.ones_like(eval_grid)
        kde = gaussian_kde(data, bw_method=KDE_BANDWIDTH / (data.std(ddof=1) + 1e-10))
        return kde.evaluate(eval_grid)