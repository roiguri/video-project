from utils import VideoProcessor
import cv2
import numpy as np
from tqdm import tqdm
import time
from scipy.stats import gaussian_kde

class BackgroundSubtractor(VideoProcessor):
    def __init__(self):
        super().__init__()
        self.BLUE_MASK_THR = 140
        self.how_many_background = 20
        self.how_many_foreground = 30
        self.max_iterations = 5
    
    def find_indices(self, src, value, how_many):
        """Find indices for a specific value in the src, returns also *how_many* points"""
        indices = np.where(src == value)
        if len(indices[0]) < how_many:
            print(f"Not enough points in src, using {len(indices[0])} points instead of {how_many}")
            how_many = len(indices[0])
        if how_many > 0:
            indices_shuffle = np.random.choice(len(indices[0]), how_many)
            return np.column_stack((indices[0][indices_shuffle], indices[1][indices_shuffle])), how_many
        else:
            return np.array([]).reshape(0, 2), 0
    
    def check_if_new(self, dictionary, value, pdf):
        """Checking if the value already exists in the dict"""
        if value in dictionary:
            return dictionary[value]
        else:
            dictionary[value] = pdf(value)[0]
            return dictionary[value]
    
    def get_initial_mask(self, frames):
        """Create initial mask using KNN background subtractor"""
        fgbg = cv2.createBackgroundSubtractorKNN()
        frames_mask = []
        
        for iteration in range(self.max_iterations):
            iteration_masks = []
            
            for frame in tqdm(frames, desc=f"KNN iteration {iteration + 1}/{self.max_iterations}", leave=False, ncols=80):
                # Convert to HSV and use only Saturation and Value channels
                transformed_frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_sv = transformed_frame_HSV[:, :, 1:]  # S and V channels
                
                # Apply background subtractor
                fgMask = fgbg.apply(frame_sv)
                fgMask = (fgMask > 130).astype(np.uint8)
                iteration_masks.append(fgMask)
            
            frames_mask = iteration_masks
        
        return frames_mask
    
    def improve_mask(self, frames, frames_mask):
        """Improve masks using morphological operations and blue channel filtering"""
        frame_count = len(frames)
        background_values = np.empty((self.how_many_background * frame_count, 3))
        foreground_values = np.empty((self.how_many_foreground * frame_count, 3))
        
        start_foreground = 0
        start_background = 0
        improved_masks = []
        
        for index_frame, frame in enumerate(tqdm(frames, desc="Improving masks", leave=False, ncols=80)):
            mask = frames_mask[index_frame]
            blue_frame, _, _ = cv2.split(frame)
            
            # Morphological operations to clean and restore
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours and get largest one
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                person_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
            else:
                person_mask = np.zeros(mask.shape, dtype=np.uint8)
            
            # Apply blue channel filter
            blue_mask = (blue_frame < self.BLUE_MASK_THR).astype(np.uint8)
            person_mask = (person_mask * blue_mask).astype(np.uint8)
            improved_masks.append(person_mask)
            
            # Collect foreground and background samples
            indices_foreground, actual_fg = self.find_indices(person_mask, 1, self.how_many_foreground)
            indices_background, actual_bg = self.find_indices(person_mask, 0, self.how_many_background)
            
            if actual_fg > 0:
                foreground_values[start_foreground:start_foreground + actual_fg] = frame[indices_foreground[:, 0], indices_foreground[:, 1], :]
                start_foreground += actual_fg
            
            if actual_bg > 0:
                background_values[start_background:start_background + actual_bg] = frame[indices_background[:, 0], indices_background[:, 1], :]
                start_background += actual_bg
        
        # Trim arrays to actual size
        background_values = background_values[:start_background]
        foreground_values = foreground_values[:start_foreground]
        
        return improved_masks, background_values, foreground_values
    
    def get_binary_and_extracted_frames(self, frames, frames_mask, background_values, foreground_values):
        """Generate final binary and extracted frames using KDE like reference implementation"""
        # Create KDE probability distributions exactly like reference
        pdf_foreground = gaussian_kde(np.asarray(foreground_values).T, bw_method=0.95)
        pdf_background = gaussian_kde(np.asarray(background_values).T, bw_method=0.95)
        
        # Memoization dictionaries to avoid recalculating same values
        pdf_foreground_dict = dict()
        pdf_background_dict = dict()
        
        binary_frames = []
        extracted_frames = []
        
        for index_frame, frame in enumerate(tqdm(frames, desc="Generating frames with KDE", leave=False, ncols=80)):
            mask = frames_mask[index_frame].copy()
            new_mask = np.zeros_like(mask)
            positions = np.where(mask == 1)
            
            # Check probability of each pixel in the mask
            check_probability_foreground = np.fromiter(
                map(lambda elem: self.check_if_new(pdf_foreground_dict, elem, pdf_foreground),
                    map(tuple, frame[positions])), dtype=float)
            check_probability_background = np.fromiter(
                map(lambda elem: self.check_if_new(pdf_background_dict, elem, pdf_background),
                    map(tuple, frame[positions])), dtype=float)
            new_mask[positions] = (check_probability_foreground > check_probability_background).astype(np.uint8)
            
            # Apply morphological operations like reference
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_mask = cv2.erode(new_mask, kernel).astype(np.uint8)
            contours, _ = cv2.findContours(new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                person_mask = np.zeros((new_mask.shape), dtype=np.uint8)
                cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
                person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, np.ones((15, 15))).astype(np.uint8)
            else:
                person_mask = new_mask
            
            # Convert to 255 for binary
            person_mask[person_mask == 1] = 255
            binary_frames.append(person_mask)
            extracted_frames.append(cv2.bitwise_and(frame, frame, mask=person_mask))
        
        return extracted_frames, binary_frames
    
    def subtract_background(self, input_video_path, background_img_path, extracted_output_path, binary_output_path):
        """Main background subtraction function using the reference implementation approach"""
        start_time = time.time()
        
        # Set random seed for reproducibility
        np.random.seed(0)
        
        # Read video frames
        frames, metadata = self.read_video(input_video_path)
        
        if not frames:
            raise ValueError(f"No frames found in {input_video_path}")
        
        # Step 1: Create initial masks
        initial_masks = self.get_initial_mask(frames)
        
        # Step 2: Improve masks
        improved_masks, background_values, foreground_values = self.improve_mask(frames, initial_masks)
        
        # Step 3: Generate final frames
        extracted_frames, binary_frames = self.get_binary_and_extracted_frames(
            frames, improved_masks, background_values, foreground_values)
        
        # Save output videos
        self.write_video(extracted_frames, extracted_output_path, metadata['fps'])
        self.write_video(binary_frames, binary_output_path, metadata['fps'])
        
        return binary_frames, extracted_frames