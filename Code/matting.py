from utils import VideoProcessor
import cv2
import numpy as np
from tqdm import tqdm
import time

class VideoMatter(VideoProcessor):
    def __init__(self):
        super().__init__()
        
        # Trimap generation parameters
        self.ERODE_SIZE = 5
        self.DILATE_SIZE = 15
        
        # Alpha matting parameters
        self.BLUR_KERNEL_SIZE = 5
        self.EDGE_THRESHOLD = 50
    
    def generate_trimap(self, binary_mask):
        """Generate trimap from binary mask
        Returns: trimap with values 0 (background), 128 (unknown), 255 (foreground)
        """
        # Ensure binary mask is single channel
        if len(binary_mask.shape) == 3:
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0-255 range
        binary_mask = (binary_mask > 127).astype(np.uint8) * 255
        
        # Create foreground region (eroded mask)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (self.ERODE_SIZE, self.ERODE_SIZE))
        foreground = cv2.erode(binary_mask, kernel_erode, iterations=1)
        
        # Create background region (dilated inverted mask)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (self.DILATE_SIZE, self.DILATE_SIZE))
        background_mask = cv2.dilate(binary_mask, kernel_dilate, iterations=1)
        background = 255 - background_mask
        
        # Create trimap
        trimap = np.full(binary_mask.shape, 128, dtype=np.uint8)  # Unknown region
        trimap[foreground == 255] = 255  # Foreground
        trimap[background == 255] = 0    # Background
        
        return trimap
    
    def fast_guided_filter(self, guide, input_img, radius=8, epsilon=0.01):
        """Fast implementation of guided filter for alpha refinement"""
        # Box filter implementation using OpenCV
        def box_filter(img, r):
            return cv2.boxFilter(img, -1, (2*r+1, 2*r+1), normalize=True)
        
        # Convert to float
        guide = guide.astype(np.float64)
        input_img = input_img.astype(np.float64)
        
        # Step 1: Compute mean
        mean_I = box_filter(guide, radius)
        mean_p = box_filter(input_img, radius)
        
        # Step 2: Compute correlation and covariance
        corr_Ip = box_filter(guide * input_img, radius)
        cov_Ip = corr_Ip - mean_I * mean_p
        
        mean_II = box_filter(guide * guide, radius)
        var_I = mean_II - mean_I * mean_I
        
        # Step 3: Compute coefficients
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        
        # Step 4: Compute mean of coefficients
        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)
        
        # Step 5: Compute output
        return mean_a * guide + mean_b
    
    def solve_alpha_matting(self, image, trimap):
        """Fast alpha matting using edge-aware smoothing"""
        # Convert trimap to initial alpha
        alpha = trimap.astype(np.float64) / 255.0
        
        # Find unknown regions
        unknown_mask = (trimap == 128)
        
        if not np.any(unknown_mask):
            return trimap
        
        # Convert image to grayscale for edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply guided filter for edge-aware smoothing
        gray_norm = gray.astype(np.float64) / 255.0
        alpha_refined = self.fast_guided_filter(gray_norm, alpha, radius=4, epsilon=0.01)
        
        # Apply bilateral filter for additional smoothing in unknown regions
        alpha_bilateral = cv2.bilateralFilter(
            alpha_refined.astype(np.float32), 9, 75, 75
        ).astype(np.float64)
        
        # Combine refined alpha with original known values
        alpha_final = alpha.copy()
        alpha_final[unknown_mask] = alpha_bilateral[unknown_mask]
        
        # Additional edge refinement using image gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_mask = edge_magnitude > self.EDGE_THRESHOLD
        
        # Sharp transitions at strong edges
        strong_edges = edge_mask & unknown_mask
        alpha_final[strong_edges] = np.where(
            alpha_final[strong_edges] > 0.5, 1.0, 0.0
        )
        
        # Clamp and convert to uint8
        alpha_final = np.clip(alpha_final, 0, 1)
        return (alpha_final * 255).astype(np.uint8)
    
    def composite_with_background(self, foreground, alpha, background):
        """Composite foreground with background using alpha channel"""
        # Normalize alpha to [0, 1]
        alpha_norm = alpha.astype(np.float64) / 255.0
        
        # Ensure all images have same dimensions
        h, w = foreground.shape[:2]
        background = cv2.resize(background, (w, h))
        
        # Convert to float for blending
        fg = foreground.astype(np.float64)
        bg = background.astype(np.float64)
        
        # Expand alpha to 3 channels if needed
        if len(fg.shape) == 3:
            alpha_3ch = np.stack([alpha_norm] * 3, axis=2)
        else:
            alpha_3ch = alpha_norm
        
        # Alpha blending: result = alpha * fg + (1 - alpha) * bg
        result = alpha_3ch * fg + (1 - alpha_3ch) * bg
        
        return result.astype(np.uint8)
    
    def apply_matting(self, extracted_path: str, binary_path: str, 
                     background_path: str, matted_path: str, alpha_path: str):
        """Apply streaming image matting to avoid memory issues"""
        print("=== Streaming Image Matting Started ===")
        start_time = time.time()
        
        # Open video captures
        cap_extracted = cv2.VideoCapture(extracted_path)
        cap_binary = cv2.VideoCapture(binary_path)
        
        # Get video properties
        fps = cap_extracted.get(cv2.CAP_PROP_FPS)
        width = int(cap_extracted.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_extracted.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_extracted.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read and resize background image
        background = cv2.imread(background_path)
        if background is None:
            raise ValueError(f"Could not read background image: {background_path}")
        background = cv2.resize(background, (width, height))
        
        print(f"Processing {total_frames} frames for matting (streaming mode)...")
        
        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        matted_writer = cv2.VideoWriter(matted_path, fourcc, fps, (width, height))
        alpha_writer = cv2.VideoWriter(alpha_path, fourcc, fps, (width, height))
        
        # Process frames one by one
        for frame_idx in range(total_frames):
            # Read one frame at a time
            ret1, extracted_frame = cap_extracted.read()
            ret2, binary_frame = cap_binary.read()
            
            if not ret1 or not ret2:
                break
            
            # Generate trimap from binary mask
            trimap = self.generate_trimap(binary_frame)
            
            # Solve for alpha channel
            alpha = self.solve_alpha_matting(extracted_frame, trimap)
            
            # Composite with new background
            matted_frame = self.composite_with_background(extracted_frame, alpha, background)
            
            # Write frames immediately
            matted_writer.write(matted_frame)
            alpha_3ch = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
            alpha_writer.write(alpha_3ch)
            
            # Clear variables immediately
            del trimap, alpha, matted_frame, alpha_3ch, extracted_frame, binary_frame
            
            # Progress update
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx+1}/{total_frames} frames")
            
            # Force garbage collection every 20 frames
            if frame_idx % 20 == 0:
                import gc
                gc.collect()
        
        # Cleanup
        cap_extracted.release()
        cap_binary.release()
        matted_writer.release()
        alpha_writer.release()
        
        total_time = time.time() - start_time
        print(f"=== Image Matting Complete! Total time: {total_time:.2f}s ===")
        print(f"Matted video saved: {matted_path}")
        print(f"Alpha channel saved: {alpha_path}") 