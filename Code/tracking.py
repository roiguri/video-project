from utils import VideoUtils
import json
import os
import cv2
import numpy as np
from tqdm import tqdm

# Tracking parameters
N_PARTICLES = 200
HISTOGRAM_BINS = 16
MIN_SIZE_RATIO = 0.8
MAX_SIZE_RATIO = 1.5
MAX_CHANGE_RATE = 0.10
RESAMPLE_THRESHOLD = 0.5

# Default and fallback parameters
DEFAULT_DETECTION = [100, 100, 50, 100]  # [x_center, y_center, half_width, half_height]
EDGE_THRESHOLD = 10
RESET_PARTICLE_RATIO = 0.3
RESET_NOISE_STD = 100
MIN_HALF_WIDTH = 10
MIN_HALF_HEIGHT = 20

class PersonTracker(VideoUtils):
    def __init__(self):
        super().__init__()
        self.initial_half_width = None
        self.initial_half_height = None
    def apply_tracking(self, input_path: str, binary_path: str, output_path: str, utils):
        """Track person using particle filter on matted video"""
        
        # Read matted video frames
        matted_frames, metadata = self.read_video(input_path)
        
        if not matted_frames:
            raise ValueError(f"No frames found in {input_path}")
        
        # Get initial bounding box from binary video (first frame only)
        initial_detection = self.get_initial_detection_from_binary(binary_path)
        
        # Store initial dimensions for size constraints
        self.initial_half_width = initial_detection[2]
        self.initial_half_height = initial_detection[3]
        
        # Initialize particle filter
        particles = self.initialize_particles(initial_detection, N_PARTICLES)
        
        # Get initial color histogram from first frame
        target_histogram = self.compute_normalized_histogram(matted_frames[0], initial_detection)
        
        tracking_results = {}
        output_frames = []
        
        # Process each frame
        for frame_idx in tqdm(range(len(matted_frames)), desc="Particle filter tracking", leave=False, ncols=80):
            frame = matted_frames[frame_idx]
            
            if frame_idx == 0:
                # First frame: use initial detection
                current_detection = initial_detection
                weights = np.ones(N_PARTICLES) / N_PARTICLES
            else:
                # Predict particles (motion model + noise)
                particles = self.predict_particles(particles)
                
                # Compute weights based on color similarity
                weights = self.compute_weights(frame, particles, target_histogram)
                
                # Get best estimate (weighted average)
                current_detection = self.get_best_estimate(particles, weights)
                
                # Update target histogram every frame
                target_histogram = self.compute_normalized_histogram(frame, current_detection)
                
                # Fallback: if tracking drifts too far from reasonable bounds, reset some particles
                frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
                if (current_detection[0] < EDGE_THRESHOLD or current_detection[0] > frame.shape[1] - EDGE_THRESHOLD or 
                    current_detection[1] < EDGE_THRESHOLD or current_detection[1] > frame.shape[0] - EDGE_THRESHOLD):
                    # Reset particles to center area
                    reset_count = int(RESET_PARTICLE_RATIO * particles.shape[1])
                    particles[0, :reset_count] = (frame_center_x + np.random.normal(0, RESET_NOISE_STD, reset_count)).astype(int)
                    particles[1, :reset_count] = (frame_center_y + np.random.normal(0, RESET_NOISE_STD, reset_count)).astype(int)
                
                # Resample particles
                particles = self.resample_particles(particles, weights)
            
            # Store tracking result in format [ROW, COL, HEIGHT, WIDTH]
            x_center, y_center, half_width, half_height = current_detection
            row = y_center - half_height  # top-left y
            col = x_center - half_width   # top-left x
            height = 2 * half_height
            width = 2 * half_width
            tracking_results[str(frame_idx)] = [int(row), int(col), int(height), int(width)]
            
            # Draw tracking results
            frame_with_tracking = self.draw_tracking_results(frame, current_detection)
            output_frames.append(frame_with_tracking)
        
        # Save output video
        self.write_video(output_frames, output_path, metadata['fps'])
        utils.record_timing('time_to_output')
        
        
        return tracking_results
    
    def get_initial_detection_from_binary(self, binary_video_path):
        """Get initial detection from first frame of binary video"""
        cap = cv2.VideoCapture(binary_video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret or first_frame is None:
            return DEFAULT_DETECTION
        if len(first_frame.shape) == 3:
            gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = first_frame
        
        # Find white pixels (person)
        indices = np.argwhere(gray_frame == 255)
        
        if len(indices) == 0:
            return DEFAULT_DETECTION
        
        # Get bounding box coordinates
        min_y, max_y = np.min(indices[:, 0]), np.max(indices[:, 0])
        min_x, max_x = np.min(indices[:, 1]), np.max(indices[:, 1])
        
        # Reduce bounding box size by 15% width and 10% height
        width = max_x - min_x
        height = max_y - min_y
        width_reduction = int(width * 0.15)
        height_reduction = int(height * 0.1)
        
        min_x += width_reduction
        max_x -= width_reduction
        min_y += height_reduction
        max_y -= height_reduction

        # Convert to center + half dimensions format
        x_center = (min_x + max_x) // 2
        y_center = (min_y + max_y) // 2
        half_width = (max_x - min_x) // 2
        half_height = (max_y - min_y) // 2
        
        return [x_center, y_center, half_width, half_height]
    
    def initialize_particles(self, initial_detection, n_particles):
        """Initialize particles around initial detection"""
        x_center, y_center, half_width, half_height = initial_detection
        
        # State: [x_center, y_center, half_width, half_height, velocity_x, velocity_y]
        particles = np.zeros((6, n_particles))
        
        # Initialize positions with small noise around detection
        particles[0, :] = x_center + np.random.normal(0, 2, n_particles)  # x_center
        particles[1, :] = y_center + np.random.normal(0, 10, n_particles)  # y_center
        particles[2, :] = half_width + np.random.normal(0, 5, n_particles)  # half_width
        particles[3, :] = half_height + np.random.normal(0, 5, n_particles)  # half_height
        particles[4, :] = np.random.normal(0, 2, n_particles)  # velocity_x
        particles[5, :] = np.random.normal(0, 2, n_particles)  # velocity_y
        
        # Ensure positive dimensions
        particles = self.ensure_minimum_bounds(particles)
        
        return particles.astype(int)
    
    def predict_particles(self, particles):
        """Apply motion model and add noise to particles"""
        particles = particles.astype(float)
        
        # Apply motion model
        particles[0, :] += particles[4, :]  # x = x + v_x
        particles[1, :] += particles[5, :]  # y = y + v_y
        
        # Store original dimensions for constraint checking
        original_half_width = particles[2, :].copy()
        original_half_height = particles[3, :].copy()
        
        # Add Gaussian noise with very small dimension noise
        noise_std = np.array([[1], [9], [1], [1], [8], [1]])  # small dimension noise
        noise = np.random.normal(0, noise_std, particles.shape)
        particles = particles + noise
        
        # Apply smart size constraints to prevent excessive shrinking/growing
        min_half_width = int(MIN_SIZE_RATIO * self.initial_half_width)
        max_half_width = int(MAX_SIZE_RATIO * self.initial_half_width)
        min_half_height = int(MIN_SIZE_RATIO * self.initial_half_height)
        max_half_height = int(MAX_SIZE_RATIO * self.initial_half_height)
        
        # Limit change rate to 10% per frame for stability
        max_width_change = MAX_CHANGE_RATE * original_half_width
        max_height_change = MAX_CHANGE_RATE * original_half_height
        
        particles[2, :] = np.clip(particles[2, :],
                                 original_half_width - max_width_change,
                                 original_half_width + max_width_change)
        particles[3, :] = np.clip(particles[3, :],
                                 original_half_height - max_height_change,
                                 original_half_height + max_height_change)
        
        # Apply absolute size constraints
        particles[2, :] = np.clip(particles[2, :], min_half_width, max_half_width)
        particles[3, :] = np.clip(particles[3, :], min_half_height, max_half_height)
        particles[0, :] = np.maximum(particles[0, :], particles[2, :])
        particles[1, :] = np.maximum(particles[1, :], particles[3, :])
        
        return particles.astype(int)
    
    def compute_normalized_histogram(self, image, detection):
        """Compute normalized color histogram for a detection"""
        x_center, y_center, half_width, half_height = detection
        
        # Extract region
        x1 = max(0, int(x_center - half_width))
        y1 = max(0, int(y_center - half_height))
        x2 = min(image.shape[1], int(x_center + half_width))
        y2 = min(image.shape[0], int(y_center + half_height))
        
        crop = image[y1:y2, x1:x2]
        
        # Compute histogram
        hist = np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS, HISTOGRAM_BINS))
        
        if crop.size > 0:
            # Quantize colors to bins per channel
            bin_size = 256 // HISTOGRAM_BINS
            max_bin = HISTOGRAM_BINS - 1
            r = np.clip(crop[:, :, 0] // bin_size, 0, max_bin)
            g = np.clip(crop[:, :, 1] // bin_size, 0, max_bin)
            b = np.clip(crop[:, :, 2] // bin_size, 0, max_bin)
            
            # Vectorized histogram computation
            r_flat = r.flatten().astype(np.int32)
            g_flat = g.flatten().astype(np.int32)
            b_flat = b.flatten().astype(np.int32)
            indices = r_flat * (HISTOGRAM_BINS ** 2) + g_flat * HISTOGRAM_BINS + b_flat
            hist_flat = np.bincount(indices, minlength=HISTOGRAM_BINS ** 3)
            hist = hist_flat.reshape(HISTOGRAM_BINS, HISTOGRAM_BINS, HISTOGRAM_BINS)
        
        # Flatten and normalize
        hist = hist.flatten()
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            hist = np.ones_like(hist) / len(hist)
        
        return hist
    
    def bhattacharyya_distance(self, p, q):
        """Compute Bhattacharyya distance between histograms"""
        return np.exp(20 * np.sum(np.sqrt(p * q)))
    
    def compute_weights(self, frame, particles, target_histogram):
        """Compute particle weights based on color similarity (reduced influence)"""
        weights = np.zeros(particles.shape[1])
        
        for i in range(particles.shape[1]):
            particle_histogram = self.compute_normalized_histogram(frame, particles[:4, i])
            color_similarity = self.bhattacharyya_distance(particle_histogram, target_histogram)
            weights[i] = color_similarity ** 0.5
        
        # Normalize weights
        weight_sum = np.sum(weights)
        weights = weights / weight_sum
        
        return weights
    
    def get_best_estimate(self, particles, weights):
        """Get best state estimate (weighted average)"""
        weighted_state = np.average(particles, axis=1, weights=weights)
        
        # Convert to bbox format
        return [int(weighted_state[0]), int(weighted_state[1]),
                int(weighted_state[2]), int(weighted_state[3])]
    
    def resample_particles(self, particles, weights):
        """Resample particles based on weights with diversity preservation"""
        n_particles = particles.shape[1]
        
        # Check if resampling is needed
        effective_size = 1.0 / np.sum(weights**2)
        
        if effective_size > n_particles * RESAMPLE_THRESHOLD:
            return particles
        
        # Compute CDF
        cdf = np.cumsum(weights)
        cdf = cdf / cdf[-1]
        
        # Sample particles
        rand_vals = np.random.rand(n_particles)
        indices = np.searchsorted(cdf, rand_vals)
        
        # Create new particle set
        resampled_particles = particles[:, indices].copy()
        
        # Add small noise to resampled particles to maintain diversity
        diversity_noise = np.random.normal(0, 2, resampled_particles.shape)
        resampled_particles = resampled_particles + diversity_noise
        
        # Ensure minimum bounds
        resampled_particles = self.ensure_minimum_bounds(resampled_particles)
        
        return resampled_particles.astype(int)
    
    def draw_tracking_results(self, frame, person_location):
        """Draw tracking results on frame"""
        frame_with_tracking = frame.copy()
        
        # Draw best estimate as green rectangle
        x_center, y_center, half_width, half_height = person_location
        x1, y1 = x_center - half_width, y_center - half_height
        x2, y2 = x_center + half_width, y_center + half_height
        
        cv2.rectangle(frame_with_tracking, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        return frame_with_tracking
    
    def save_tracking_json(self, output_dir: str, tracking_results: dict):
        """Save tracking results to JSON"""
        tracking_path = os.path.join(output_dir, 'tracking.json')
        with open(tracking_path, 'w') as f:
            json.dump(tracking_results, f, indent=2)
        print(f"Tracking data saved to {tracking_path}")
    
    def ensure_minimum_bounds(self, particles):
        """Helper function for minimum bounds checking"""
        particles[2, :] = np.maximum(particles[2, :], MIN_HALF_WIDTH)
        particles[3, :] = np.maximum(particles[3, :], MIN_HALF_HEIGHT)
        return particles