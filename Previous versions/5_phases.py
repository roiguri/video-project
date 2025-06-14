import cv2
import numpy as np
from tqdm import tqdm
from utils import VideoProcessor
import time
from scipy.stats import gaussian_kde

class BackgroundSubtractor(VideoProcessor):
    def __init__(self):
        super().__init__()
        
        # ========== ALL CONFIGURABLE PARAMETERS ==========
        
        # KDE Parameters
        self.BW_MEDIUM = 1
        self.BW_NARROW = 0.1
        
        # Frame dimensions and region heights
        self.LEGS_HEIGHT = 805
        self.SHOES_HEIGHT = 870
        self.SHOULDERS_HEIGHT = 405
        self.BLUE_MASK_THR = 140
        
        # Window sizes for processing
        self.WINDOW_WIDTH = 500
        self.WINDOW_HEIGHT = 1000
        self.FACE_WINDOW_HEIGHT = 250
        self.FACE_WINDOW_WIDTH = 300
        
        # Pixel sampling
        self.NUM_PIXEL = 10
        
        # Background subtractor parameters
        self.KNN_HISTORY_MULTIPLIER = 3
        self.DETECT_SHADOWS = False
        
        # Morphological operations
        self.CLOSE_KERNEL_SIZE = 6
        self.MEDIAN_BLUR_SIZE = 7
        self.MORPH_OPEN_KERNEL_V = (20, 1)
        self.MORPH_OPEN_KERNEL_H = (1, 20)
        self.MORPH_CLOSE_SIZE = (1, 20)
        self.MORPH_CLOSE_DISK_SIZE = 20
        
        # Face processing parameters
        self.FACE_MORPH_KERNEL = (6, 1)
        self.FINAL_MORPH_KERNEL = (1, 6)
        self.FACE_DISK_KERNEL_SIZE = 12
        self.FACE_DILATE_KERNEL_SIZE = 3
        self.FACE_DILATE_ITERATIONS = 1
        self.FACE_BOTTOM_HEIGHT = 50
        
        # Shoes restoration parameters
        self.SHOES_DELTA_Y = 30
        self.SHOES_THRESHOLD = 0.75
        self.SHOES_RESTORE_HEIGHT = 270
        
        # Processing control parameters
        self.ENABLED_PHASES = [1, 2, 3, 4, 5]  # List of phases to execute
        # Phase 1: KNN background learning
        # Phase 2: Color sampling for body/shoes 
        # Phase 3: KDE statistical filtering
        # Phase 4: Face color sampling
        # Phase 5: Face refinement & final output
        
        # ================================================

    def set_enabled_phases(self, phases):
        """Set which specific phases to execute
        
        Args:
            phases: List of phase numbers to execute [1-5], or single int
                1 = KNN background learning
                2 = Color sampling for body/shoes  
                3 = KDE statistical filtering
                4 = Face color sampling
                5 = Face refinement & final output
        
        Examples:
            set_enabled_phases([1, 3, 5])  # Skip color sampling phases
            set_enabled_phases([1, 2])     # Basic processing only
            set_enabled_phases(3)          # Only statistical filtering
        """
        if isinstance(phases, int):
            phases = [phases]
        
        if not all(1 <= p <= 5 for p in phases):
            raise ValueError("All phases must be between 1 and 5")
        
        self.ENABLED_PHASES = sorted(phases)
        print(f"Enabled phases: {self.ENABLED_PHASES}")
    
    def set_max_phases(self, num_phases: int):
        """Set the maximum number of phases to execute (1-5) - convenience method
        
        Args:
            num_phases: Number of phases to execute sequentially from 1
        """
        if not 1 <= num_phases <= 5:
            raise ValueError("num_phases must be between 1 and 5")
        self.ENABLED_PHASES = list(range(1, num_phases + 1))
        print(f"Set processing to {num_phases} phases: {self.ENABLED_PHASES}")

    def extract_person(self, input_path: str, extracted_path: str, binary_path: str):
        """Extract person using advanced KDE-based background subtraction with memory optimization"""
        print("=== Advanced Background Subtraction Started ===")
        start_time = time.time()
        
        # Get video metadata without loading all frames
        cap = cv2.VideoCapture(input_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"Processing {n_frames} frames ({w}x{h}) at {fps} FPS")
        print(f"Executing phases: {self.ENABLED_PHASES}")
        
        # Phase 1: Background Subtractor KNN studying - stream frames
        if 1 in self.ENABLED_PHASES:
            print("[BS] - BackgroundSubtractorKNN Studying Frames history")
            backSub = cv2.createBackgroundSubtractorKNN(
                detectShadows=self.DETECT_SHADOWS, 
                history=self.KNN_HISTORY_MULTIPLIER * n_frames
            )
            mask_list = self._background_subtractor_knn_studying_streaming(input_path, backSub, n_frames, h, w)
            print(f"[BS] - BackgroundSubtractorKNN Finished at: {time.time() - start_time:.2f}s")
        else:
            print("Skipping Phase 1 - Creating empty masks")
            mask_list = np.zeros((n_frames, h, w), dtype=np.uint8)
        
        # Phase 2: Collect colors for body and shoes KDEs - stream frames
        if 2 in self.ENABLED_PHASES:
            print("Collecting colors for building body & shoes KDEs")
            start_collecting = time.time()
            (person_and_blue_mask_list, 
             omega_f_colors, omega_b_colors, 
             omega_f_shoes_colors, omega_b_shoes_colors) = self._collecting_colors_body_and_shoes_streaming(
                input_path, mask_list, n_frames, h, w
            )
            print(f"Collecting colors completed at: {time.time() - start_collecting:.2f}s")
        else:
            print("Skipping Phase 2 - Using Phase 1 masks as final masks")
            person_and_blue_mask_list = mask_list
            omega_f_colors = np.empty((0, 3))
            omega_b_colors = np.empty((0, 3))
            omega_f_shoes_colors = np.empty((0, 3))
            omega_b_shoes_colors = np.empty((0, 3))
        
        # Phase 3: Filter using KDEs for general body parts and shoes - stream frames
        if 3 in self.ENABLED_PHASES:
            print("Filtering with KDEs general body parts & shoes")
            start_filtering = time.time()
            or_mask_list = self._filtering_kdes_memory_optimized(
                omega_f_colors, omega_b_colors, omega_f_shoes_colors, omega_b_shoes_colors,
                input_path, person_and_blue_mask_list, n_frames, h, w
            )
            print(f"KDE filtering completed at: {time.time() - start_filtering:.2f}s")
        else:
            print("Skipping Phase 3 - Using Phase 2 masks as final masks")
            or_mask_list = person_and_blue_mask_list
        
        # Phase 4: Collect colors for face processing - stream frames
        if 4 in self.ENABLED_PHASES:
            print("Collecting colors for face")
            start_face_colors = time.time()
            omega_f_face_colors, omega_b_face_colors = self._collecting_colors_face_streaming(
                input_path, or_mask_list, h, w
            )
            print(f"Face color collection completed at: {time.time() - start_face_colors:.2f}s")
        else:
            print("Skipping Phase 4 - No face color collection")
            omega_f_face_colors = np.empty((0, 3))
            omega_b_face_colors = np.empty((0, 3))
        
        # Phase 5: Final processing with face refinement - stream frames and write directly
        if 5 in self.ENABLED_PHASES:
            print("Final Processing with face refinement")
            start_final = time.time()
            self._final_processing_streaming(
                omega_f_face_colors, omega_b_face_colors, input_path, or_mask_list, 
                h, w, extracted_path, binary_path, fps
            )
            print(f"Final processing completed at: {time.time() - start_final:.2f}s")
        else:
            print("Skipping Phase 5 - Creating basic output from available masks")
            self._create_basic_output(input_path, or_mask_list, extracted_path, binary_path, fps)
        
        # Cleanup mask lists to free memory
        del mask_list, person_and_blue_mask_list, or_mask_list
        
        total_time = time.time() - start_time
        print(f"=== Background Subtraction Complete! Total time: {total_time:.2f}s ===")
        print(f"Extracted video saved: {extracted_path}")
        print(f"Binary mask saved: {binary_path}")

    def _background_subtractor_knn_studying_streaming(self, input_path, backSub, n_frames, h, w):
        """Study frames using BackgroundSubtractorKNN with streaming to reduce memory"""
        mask_list = np.zeros((n_frames, h, w), dtype=np.uint8)
        
        for _ in tqdm(range(6), desc="BackgroundSubtractorKNN Studying Frames history"):
            cap = cv2.VideoCapture(input_path)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_sv = frame_hsv[:, :, 1:]  # Use saturation and value channels
                fg_mask = backSub.apply(frame_sv)
                fg_mask = (fg_mask > 200).astype(np.uint8)
                mask_list[frame_idx] = fg_mask
                frame_idx += 1
            
            cap.release()
        
        return mask_list

    def _collecting_colors_body_and_shoes_streaming(self, input_path, mask_list, n_frames, h, w):
        """Collect color samples for body and shoes KDE estimation with streaming"""
        # Limit color sample accumulation to prevent memory explosion
        max_samples_per_type = 10000  # Limit total samples to keep memory reasonable while ensuring enough data
        omega_f_colors_list, omega_b_colors_list = [], []
        omega_f_shoes_colors_list, omega_b_shoes_colors_list = [], []
        person_and_blue_mask_list = np.zeros((n_frames, h, w), dtype=np.uint8)
        
        cap = cv2.VideoCapture(input_path)
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            blue_frame, _, _ = cv2.split(frame)
            mask_for_frame = mask_list[frame_index].astype(np.uint8)
            
            # Apply morphological operations
            mask_for_frame = cv2.morphologyEx(mask_for_frame, cv2.MORPH_CLOSE, 
                                            self._disk_kernel(self.CLOSE_KERNEL_SIZE))
            mask_for_frame = cv2.medianBlur(mask_for_frame, self.MEDIAN_BLUR_SIZE)
            
            # Find largest contour (person)
            contours, _ = cv2.findContours(mask_for_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = list(contours)
                contours.sort(key=cv2.contourArea, reverse=True)
            
            person_mask = np.zeros(mask_for_frame.shape)
            if contours:
                cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
            
            # Create blue mask (avoid blue screen areas)
            blue_mask = (blue_frame < self.BLUE_MASK_THR).astype(np.uint8)
            person_and_blue_mask = (person_mask * blue_mask).astype(np.uint8)
            
            # Sample foreground and background pixels
            omega_f_indices = self._choose_indices_for_background_and_foreground(
                person_and_blue_mask, self.NUM_PIXEL, 1)
            omega_b_indices = self._choose_indices_for_background_and_foreground(
                person_and_blue_mask, self.NUM_PIXEL, 0)
            
            # Sample shoes area specifically
            shoes_mask = np.copy(person_and_blue_mask)
            shoes_mask[:self.SHOES_HEIGHT, :] = 0
            omega_f_shoes_indices = self._choose_indices_for_background_and_foreground(
                shoes_mask, self.NUM_PIXEL, 1)
            
            shoes_mask = np.copy(person_and_blue_mask)
            shoes_mask[:self.SHOES_HEIGHT - 120, :] = 1
            omega_b_shoes_indices = self._choose_indices_for_background_and_foreground(
                shoes_mask, self.NUM_PIXEL, 0)
            
            person_and_blue_mask_list[frame_index] = person_and_blue_mask
            
            # Accumulate color samples with memory limits
            if len(omega_f_colors_list) < max_samples_per_type // self.NUM_PIXEL:
                omega_f_colors_list.append(frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :])
                omega_b_colors_list.append(frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :])
                omega_f_shoes_colors_list.append(frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :])
                omega_b_shoes_colors_list.append(frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :])
            
            frame_index += 1
        
        cap.release()
        
        # Concatenate limited samples
        omega_f_colors = np.vstack(omega_f_colors_list) if omega_f_colors_list else np.empty((0, 3))
        omega_b_colors = np.vstack(omega_b_colors_list) if omega_b_colors_list else np.empty((0, 3))
        omega_f_shoes_colors = np.vstack(omega_f_shoes_colors_list) if omega_f_shoes_colors_list else np.empty((0, 3))
        omega_b_shoes_colors = np.vstack(omega_b_shoes_colors_list) if omega_b_shoes_colors_list else np.empty((0, 3))
        
        return (person_and_blue_mask_list, omega_f_colors, omega_b_colors, omega_f_shoes_colors, omega_b_shoes_colors)

    def _filtering_kdes_memory_optimized(self, omega_f_colors, omega_b_colors, omega_f_shoes_colors, omega_b_shoes_colors,
                                        input_path, person_and_blue_mask_list, n_frames, h, w):
        """Filter using KDE probability distributions with streaming and limited memoization"""
        # Build KDE models
        foreground_pdf = self._new_estimate_pdf(omega_values=omega_f_colors, bw_method=self.BW_MEDIUM)
        background_pdf = self._new_estimate_pdf(omega_values=omega_b_colors, bw_method=self.BW_MEDIUM)
        foreground_shoes_pdf = self._new_estimate_pdf(omega_values=omega_f_shoes_colors, bw_method=self.BW_MEDIUM)
        background_shoes_pdf = self._new_estimate_pdf(omega_values=omega_b_shoes_colors, bw_method=self.BW_MEDIUM)
        
        # Limited memoization with LRU-like behavior
        max_cache_size = 10000
        foreground_pdf_memoization, background_pdf_memoization = {}, {}
        foreground_shoes_pdf_memoization, background_shoes_pdf_memoization = {}, {}
        
        or_mask_list = np.zeros((n_frames, h, w), dtype=np.uint8)
        
        cap = cv2.VideoCapture(input_path)
        frame_index = 0
        
        print(f"Processing {n_frames} frames for KDE filtering...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            person_and_blue_mask = person_and_blue_mask_list[frame_index]
            person_and_blue_mask_indices = np.where(person_and_blue_mask == 1)
            
            if len(person_and_blue_mask_indices[0]) == 0:
                frame_index += 1
                continue
                
            y_mean = int(np.mean(person_and_blue_mask_indices[0]))
            x_mean = int(np.mean(person_and_blue_mask_indices[1]))
            
            # Extract window around person
            small_frame_bgr = frame[
                max(0, y_mean - self.WINDOW_HEIGHT // 2):min(h, y_mean + self.WINDOW_HEIGHT // 2),
                max(0, x_mean - self.WINDOW_WIDTH // 2):min(w, x_mean + self.WINDOW_WIDTH // 2),
                :
            ]
            
            small_person_and_blue_mask = person_and_blue_mask[
                max(0, y_mean - self.WINDOW_HEIGHT // 2):min(h, y_mean + self.WINDOW_HEIGHT // 2),
                max(0, x_mean - self.WINDOW_WIDTH // 2):min(w, x_mean + self.WINDOW_WIDTH // 2)
            ]
            
            small_person_and_blue_mask_indices = np.where(small_person_and_blue_mask == 1)
            small_probs_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
            
            # Clear cache if too large
            if len(foreground_pdf_memoization) > max_cache_size:
                foreground_pdf_memoization.clear()
                background_pdf_memoization.clear()
            
            # Calculate probabilities for body parts - optimized batch processing
            if len(small_person_and_blue_mask_indices[0]) == 0:
                frame_index += 1
                continue
            
            # Get pixel colors for this mask region
            pixel_colors = small_frame_bgr[small_person_and_blue_mask_indices]
            
            # Limit processing to avoid hanging on large pixel sets
            if len(pixel_colors) > 5000:  # Subsample if too many pixels
                sample_indices = np.random.choice(len(pixel_colors), 5000, replace=False)
                pixel_colors = pixel_colors[sample_indices]
                sampled_indices = (small_person_and_blue_mask_indices[0][sample_indices], 
                                 small_person_and_blue_mask_indices[1][sample_indices])
            else:
                sampled_indices = small_person_and_blue_mask_indices
            
            # Batch process probabilities more efficiently
            small_foreground_probabilities = np.array([
                self._check_in_dict(foreground_pdf_memoization, tuple(color), foreground_pdf)
                for color in pixel_colors
            ])
            small_background_probabilities = np.array([
                self._check_in_dict(background_pdf_memoization, tuple(color), background_pdf) 
                for color in pixel_colors
            ])
            
            small_probs_fg_bigger_bg_mask[sampled_indices] = (
                small_foreground_probabilities > small_background_probabilities).astype(np.uint8)
            
            # Shoes restoration
            smaller_upper_white_mask = np.copy(small_probs_fg_bigger_bg_mask)
            smaller_upper_white_mask[:-self.SHOES_RESTORE_HEIGHT, :] = 1
            small_probs_fg_bigger_bg_mask_black_indices = np.where(smaller_upper_white_mask == 0)
            
            small_probs_shoes_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
            
            if len(small_probs_fg_bigger_bg_mask_black_indices[0]) > 0:
                # Clear shoes cache if too large
                if len(foreground_shoes_pdf_memoization) > max_cache_size:
                    foreground_shoes_pdf_memoization.clear()
                    background_shoes_pdf_memoization.clear()
                
                # Optimized shoes probability calculation with limits
                shoes_pixel_colors = small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices]
                
                # Limit shoes processing as well
                if len(shoes_pixel_colors) > 2000:
                    shoes_sample_indices = np.random.choice(len(shoes_pixel_colors), 2000, replace=False)
                    shoes_pixel_colors = shoes_pixel_colors[shoes_sample_indices]
                    shoes_sampled_indices = (small_probs_fg_bigger_bg_mask_black_indices[0][shoes_sample_indices],
                                           small_probs_fg_bigger_bg_mask_black_indices[1][shoes_sample_indices])
                else:
                    shoes_sampled_indices = small_probs_fg_bigger_bg_mask_black_indices
                
                small_shoes_foreground_probabilities = np.array([
                    self._check_in_dict(foreground_shoes_pdf_memoization, tuple(color), foreground_shoes_pdf)
                    for color in shoes_pixel_colors
                ])
                small_shoes_background_probabilities = np.array([
                    self._check_in_dict(background_shoes_pdf_memoization, tuple(color), background_shoes_pdf)
                    for color in shoes_pixel_colors
                ])
                
                shoes_fg_shoes_bg_ratio = small_shoes_foreground_probabilities / (
                    small_shoes_foreground_probabilities + small_shoes_background_probabilities)
                shoes_fg_beats_shoes_bg_mask = (shoes_fg_shoes_bg_ratio > self.SHOES_THRESHOLD).astype(np.uint8)
                small_probs_shoes_fg_bigger_bg_mask[shoes_sampled_indices] = shoes_fg_beats_shoes_bg_mask
            
            # Combine body and shoes masks
            small_probs_shoes_fg_bigger_bg_mask_indices = np.where(small_probs_shoes_fg_bigger_bg_mask == 1)
            if len(small_probs_shoes_fg_bigger_bg_mask_indices[0]) > 0:
                y_shoes_mean = int(np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[0]))
            else:
                y_shoes_mean = 0
            
            small_or_mask = np.zeros(small_probs_fg_bigger_bg_mask.shape)
            small_or_mask[:y_shoes_mean, :] = small_probs_fg_bigger_bg_mask[:y_shoes_mean, :]
            small_or_mask[y_shoes_mean:, :] = np.maximum(
                small_probs_fg_bigger_bg_mask[y_shoes_mean:, :], 
                small_probs_shoes_fg_bigger_bg_mask[y_shoes_mean:, :]
            ).astype(np.uint8)
            
            # Apply morphological operations to shoes area
            small_or_mask[y_shoes_mean - self.SHOES_DELTA_Y:, :] = cv2.morphologyEx(
                small_or_mask[y_shoes_mean - self.SHOES_DELTA_Y:, :],
                cv2.MORPH_CLOSE, np.ones(self.MORPH_CLOSE_SIZE))
            small_or_mask[y_shoes_mean - self.SHOES_DELTA_Y:, :] = cv2.morphologyEx(
                small_or_mask[y_shoes_mean - self.SHOES_DELTA_Y:, :],
                cv2.MORPH_CLOSE, self._disk_kernel(self.MORPH_CLOSE_DISK_SIZE))
            
            # Map back to full frame
            or_mask = np.zeros(person_and_blue_mask.shape, dtype=np.uint8)
            or_mask[
                max(0, y_mean - self.WINDOW_HEIGHT // 2):min(h, y_mean + self.WINDOW_HEIGHT // 2),
                max(0, x_mean - self.WINDOW_WIDTH // 2):min(w, x_mean + self.WINDOW_WIDTH // 2)
            ] = small_or_mask
            
            or_mask_list[frame_index] = or_mask
            frame_index += 1
            
            # Progress tracking
            if frame_index % 50 == 0:
                print(f"KDE filtering progress: {frame_index}/{n_frames} frames ({frame_index/n_frames*100:.1f}%)")
        
        cap.release()
        return or_mask_list

    def _collecting_colors_face_streaming(self, input_path, or_mask_list, h, w):
        """Collect color samples for face KDE estimation with streaming and memory limits"""
        max_face_samples = 5000  # Limit face samples while ensuring sufficient data
        omega_f_face_colors_list, omega_b_face_colors_list = [], []
        
        cap = cv2.VideoCapture(input_path)
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            or_mask = or_mask_list[frame_index]
            face_mask = np.copy(or_mask)
            face_mask[self.SHOULDERS_HEIGHT:, :] = 0
            
            face_mask_indices = np.where(face_mask == 1)
            if len(face_mask_indices[0]) == 0:
                frame_index += 1
                continue
                
            y_mean = int(np.mean(face_mask_indices[0]))
            x_mean = int(np.mean(face_mask_indices[1]))
            
            # Extract face window
            small_frame_bgr = frame[
                max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):min(h, y_mean + self.FACE_WINDOW_HEIGHT // 2),
                max(0, x_mean - self.FACE_WINDOW_WIDTH // 2):min(w, x_mean + self.FACE_WINDOW_WIDTH // 2),
                :
            ]
            
            small_face_mask = face_mask[
                max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):min(h, y_mean + self.FACE_WINDOW_HEIGHT // 2),
                max(0, x_mean - self.FACE_WINDOW_WIDTH // 2):min(w, x_mean + self.FACE_WINDOW_WIDTH // 2)
            ]
            
            # Apply morphological operations to face mask
            small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, 
                                             np.ones(self.MORPH_OPEN_KERNEL_V, np.uint8))
            small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, 
                                             np.ones(self.MORPH_OPEN_KERNEL_H, np.uint8))
            
            # Sample face pixels with limits
            if len(omega_f_face_colors_list) < max_face_samples // 20:
                omega_f_face_indices = self._choose_indices_for_background_and_foreground(small_face_mask, 20, 1)
                omega_b_face_indices = self._choose_indices_for_background_and_foreground(small_face_mask, 20, 0)
                
                omega_f_face_colors_list.append(small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :])
                omega_b_face_colors_list.append(small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :])
            
            frame_index += 1
        
        cap.release()
        
        omega_f_face_colors = np.vstack(omega_f_face_colors_list) if omega_f_face_colors_list else np.empty((0, 3))
        omega_b_face_colors = np.vstack(omega_b_face_colors_list) if omega_b_face_colors_list else np.empty((0, 3))
        
        return omega_f_face_colors, omega_b_face_colors

    def _final_processing_streaming(self, omega_f_face_colors, omega_b_face_colors, input_path, or_mask_list, 
                                   h, w, extracted_path, binary_path, fps):
        """Final processing with face refinement, streaming and direct video writing"""
        foreground_face_pdf = self._new_estimate_pdf(omega_values=omega_f_face_colors, bw_method=self.BW_NARROW)
        background_face_pdf = self._new_estimate_pdf(omega_values=omega_b_face_colors, bw_method=self.BW_NARROW)
        
        # Limited memoization
        max_cache_size = 5000
        foreground_face_pdf_memoization, background_face_pdf_memoization = {}, {}
        
        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_writer = cv2.VideoWriter(extracted_path, fourcc, fps, (w, h))
        binary_writer = cv2.VideoWriter(binary_path, fourcc, fps, (w, h))
        
        cap = cv2.VideoCapture(input_path)
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            or_mask = or_mask_list[frame_index]
            face_mask = np.copy(or_mask)
            face_mask[self.SHOULDERS_HEIGHT:, :] = 0
            
            face_mask_indices = np.where(face_mask == 1)
            if len(face_mask_indices[0]) == 0:
                # No face detected, use original mask
                final_mask = self._scale_matrix_0_to_255(or_mask.astype(np.uint8))
                final_frame = self._apply_mask_on_color_frame(frame=frame, mask=or_mask)
                
                # Write directly to video files
                extracted_writer.write(final_frame)
                if len(final_mask.shape) == 2:
                    final_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
                else:
                    final_mask_bgr = final_mask
                binary_writer.write(final_mask_bgr)
                
                frame_index += 1
                continue
                
            y_mean = int(np.mean(face_mask_indices[0]))
            x_mean = int(np.mean(face_mask_indices[1]))
            
            # Extract face window
            small_frame_bgr = frame[
                max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):min(h, y_mean + self.FACE_WINDOW_HEIGHT // 2),
                max(0, x_mean - self.FACE_WINDOW_WIDTH // 2):min(w, x_mean + self.FACE_WINDOW_WIDTH // 2),
                :
            ]
            
            small_face_mask = face_mask[
                max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):min(h, y_mean + self.FACE_WINDOW_HEIGHT // 2),
                max(0, x_mean - self.FACE_WINDOW_WIDTH // 2):min(w, x_mean + self.FACE_WINDOW_WIDTH // 2)
            ]
            
            # Clear cache periodically
            if len(foreground_face_pdf_memoization) > max_cache_size:
                foreground_face_pdf_memoization.clear()
                background_face_pdf_memoization.clear()
            
            # Calculate face probabilities
            small_frame_bgr_stacked = small_frame_bgr.reshape((-1, 3))
            
            small_face_foreground_probabilities = np.fromiter(
                map(lambda elem: self._check_in_dict(foreground_face_pdf_memoization, elem, foreground_face_pdf),
                    map(tuple, small_frame_bgr_stacked)), dtype=float)
            small_face_background_probabilities = np.fromiter(
                map(lambda elem: self._check_in_dict(background_face_pdf_memoization, elem, background_face_pdf),
                    map(tuple, small_frame_bgr_stacked)), dtype=float)
            
            small_face_foreground_probabilities = small_face_foreground_probabilities.reshape(small_face_mask.shape)
            small_face_background_probabilities = small_face_background_probabilities.reshape(small_face_mask.shape)
            
            small_probs_face_fg_bigger_face_bg_mask = (
                small_face_foreground_probabilities > small_face_background_probabilities).astype(np.uint8)
            
            # Apply Laplacian edge detection for refinement
            small_probs_face_fg_bigger_face_bg_mask_laplacian = cv2.Laplacian(
                small_probs_face_fg_bigger_face_bg_mask, cv2.CV_32F)
            small_probs_face_fg_bigger_face_bg_mask_laplacian = np.abs(small_probs_face_fg_bigger_face_bg_mask_laplacian)
            
            small_probs_face_fg_bigger_face_bg_mask = np.maximum(
                small_probs_face_fg_bigger_face_bg_mask - small_probs_face_fg_bigger_face_bg_mask_laplacian, 0)
            small_probs_face_fg_bigger_face_bg_mask[np.where(small_probs_face_fg_bigger_face_bg_mask > 1)] = 0
            small_probs_face_fg_bigger_face_bg_mask = small_probs_face_fg_bigger_face_bg_mask.astype(np.uint8)
            
            # Find largest contour for face
            contours, _ = cv2.findContours(small_probs_face_fg_bigger_face_bg_mask, 
                                         cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contours = list(contours)
                contours.sort(key=cv2.contourArea, reverse=True)
                small_contour_mask = np.zeros(small_probs_face_fg_bigger_face_bg_mask.shape, dtype=np.uint8)
                cv2.fillPoly(small_contour_mask, pts=[contours[0]], color=1)
                
                # Apply morphological operations
                small_contour_mask = cv2.morphologyEx(small_contour_mask, cv2.MORPH_CLOSE, 
                                                    self._disk_kernel(self.FACE_DISK_KERNEL_SIZE))
                small_contour_mask = cv2.dilate(small_contour_mask, 
                                              self._disk_kernel(self.FACE_DILATE_KERNEL_SIZE), 
                                              iterations=self.FACE_DILATE_ITERATIONS).astype(np.uint8)
                
                # Preserve bottom part of original face mask
                small_contour_mask[-self.FACE_BOTTOM_HEIGHT:, :] = small_face_mask[-self.FACE_BOTTOM_HEIGHT:, :]
            else:
                small_contour_mask = small_face_mask.astype(np.uint8)
            
            # Combine with original mask
            final_mask = np.copy(or_mask).astype(np.uint8)
            final_mask[
                max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):min(h, y_mean + self.FACE_WINDOW_HEIGHT // 2),
                max(0, x_mean - self.FACE_WINDOW_WIDTH // 2):min(w, x_mean + self.FACE_WINDOW_WIDTH // 2)
            ] = small_contour_mask
            
            # Apply final morphological operations
            final_mask[max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):self.LEGS_HEIGHT, :] = cv2.morphologyEx(
                final_mask[max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):self.LEGS_HEIGHT, :], 
                cv2.MORPH_OPEN, np.ones(self.FACE_MORPH_KERNEL, np.uint8))
            final_mask[max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):self.LEGS_HEIGHT, :] = cv2.morphologyEx(
                final_mask[max(0, y_mean - self.FACE_WINDOW_HEIGHT // 2):self.LEGS_HEIGHT, :], 
                cv2.MORPH_OPEN, np.ones(self.FINAL_MORPH_KERNEL, np.uint8))
            
            # Find final largest contour
            contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = list(contours)
                contours.sort(key=cv2.contourArea, reverse=True)
                final_contour_mask = np.zeros(final_mask.shape, dtype=np.uint8)
                cv2.fillPoly(final_contour_mask, pts=[contours[0]], color=1)
                final_mask = (final_contour_mask * final_mask).astype(np.uint8)
            
            # Create final outputs and write directly
            final_mask_scaled = self._scale_matrix_0_to_255(final_mask)
            final_frame = self._apply_mask_on_color_frame(frame=frame, mask=final_mask)
            
            extracted_writer.write(final_frame)
            if len(final_mask_scaled.shape) == 2:
                final_mask_bgr = cv2.cvtColor(final_mask_scaled, cv2.COLOR_GRAY2BGR)
            else:
                final_mask_bgr = final_mask_scaled
            binary_writer.write(final_mask_bgr)
            
            frame_index += 1
        
        cap.release()
        extracted_writer.release()
        binary_writer.release()

    def _create_basic_output(self, input_path, mask_list, extracted_path, binary_path, fps):
        """Create basic output when Phase 5 is skipped"""
        cap = cv2.VideoCapture(input_path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc_color = cv2.VideoWriter_fourcc(*'MJPG')
        
        extracted_writer = cv2.VideoWriter(extracted_path, fourcc_color, fps, (w, h))
        binary_writer = cv2.VideoWriter(binary_path, fourcc, fps, (w, h))
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index < len(mask_list):
                mask = mask_list[frame_index]
                
                # Apply basic cleanup
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
                
                # Create outputs
                final_mask_scaled = self._scale_matrix_0_to_255(mask)
                final_frame = self._apply_mask_on_color_frame(frame=frame, mask=mask)
                
                extracted_writer.write(final_frame)
                if len(final_mask_scaled.shape) == 2:
                    final_mask_bgr = cv2.cvtColor(final_mask_scaled, cv2.COLOR_GRAY2BGR)
                else:
                    final_mask_bgr = final_mask_scaled
                binary_writer.write(final_mask_bgr)
            
            frame_index += 1
        
        cap.release()
        extracted_writer.release()
        binary_writer.release()

    # Helper methods
    def _choose_indices_for_background_and_foreground(self, mask, number_of_choices, b_or_f):
        """Choose random indices for foreground (1) or background (0) pixels"""
        indices = np.where(mask == b_or_f)
        if len(indices[0]) == 0:
            return np.column_stack((indices[0], indices[1]))
        if len(indices[0]) < number_of_choices:
            number_of_choices = len(indices[0])
        indices_choices = np.random.choice(len(indices[0]), number_of_choices, replace=False)
        return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))

    def _new_estimate_pdf(self, omega_values, bw_method):
        """Estimate PDF using Gaussian KDE with fallback for insufficient data"""
        if omega_values.size == 0 or len(omega_values) < 2:
            # Fallback: return uniform probability function
            return lambda x: np.ones(len(x) if hasattr(x, '__len__') else 1) * 0.001
        
        try:
            pdf = gaussian_kde(omega_values.T, bw_method=bw_method)
            return lambda x: pdf(x.T)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback for singular matrix or other KDE errors
            return lambda x: np.ones(len(x) if hasattr(x, '__len__') else 1) * 0.001

    def _check_in_dict(self, dict_cache, element, function):
        """Memoization helper for PDF calculations"""
        if element in dict_cache:
            return dict_cache[element]
        else:
            dict_cache[element] = function(np.asarray(element))[0]
            return dict_cache[element]

    def _disk_kernel(self, size):
        """Create circular morphological kernel"""
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _scale_matrix_0_to_255(self, input_matrix):
        """Scale matrix values to 0-255 range"""
        if input_matrix.dtype == np.bool:
            input_matrix = np.uint8(input_matrix)
        input_matrix = input_matrix.astype(np.uint8)
        if np.ptp(input_matrix) == 0:
            return input_matrix * 255
        scaled = 255 * (input_matrix - np.min(input_matrix)) / np.ptp(input_matrix)
        return np.uint8(scaled)

    def _apply_mask_on_color_frame(self, frame, mask):
        """Apply binary mask to color frame"""
        frame_after_mask = np.copy(frame)
        frame_after_mask[:, :, 0] = frame_after_mask[:, :, 0] * mask
        frame_after_mask[:, :, 1] = frame_after_mask[:, :, 1] * mask
        frame_after_mask[:, :, 2] = frame_after_mask[:, :, 2] * mask
        return frame_after_mask