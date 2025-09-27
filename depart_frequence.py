import numpy as np
import cv2

def depart_frequence(img, t=None):
    """
    Decompose image into low, mid, and high frequency components using frequency domain thresholding
    
    Parameters:
    img: Input image (numpy array, 0-255 range)
    t: Time step parameter (not used in this implementation)
    
    Returns:
    list: [low_frequency, mid_frequency, high_frequency] components
    """
    
    # Convert image to float32
    img_float = img.astype(np.float32)
    
    # For color images, convert to YUV and process Y channel (luminance)
    if len(img_float.shape) == 3:
        img_yuv = cv2.cvtColor(img_float.astype(np.uint8), cv2.COLOR_RGB2YUV)
        img_y = img_yuv[:,:,0].astype(np.float32)
    else:
        img_y = img_float
    
    # Get image dimensions
    h, w = img_y.shape
    
    # Apply 2D Fourier transform
    f = np.fft.fft2(img_y)
    fshift = np.fft.fftshift(f)
    
    # Create frequency coordinate grid
    u = np.arange(-w//2, w//2) if w % 2 == 0 else np.arange(-(w-1)//2, (w-1)//2+1)
    v = np.arange(-h//2, h//2) if h % 2 == 0 else np.arange(-(h-1)//2, (h-1)//2+1)
    U, V = np.meshgrid(u, v)
    
    # Calculate frequency radius
    D = np.sqrt(U**2 + V**2)
    D_max = np.max(D)
    
    # Set frequency thresholds
    low_threshold = 0.1   # 10% of maximum frequency
    mid_threshold = 0.3   # 30% of maximum frequency  
    high_threshold = 0.6  # 60% of maximum frequency
    
    # Create frequency masks
    low_mask = (D <= low_threshold * D_max).astype(np.float32)
    mid_mask = ((D > low_threshold * D_max) & (D <= mid_threshold * D_max)).astype(np.float32)
    high_mask = (D > high_threshold * D_max).astype(np.float32)
    
    # Apply Gaussian smoothing to mask edges to reduce ringing artifacts
    sigma = 2.0
    low_mask = cv2.GaussianBlur(low_mask, (0, 0), sigma)
    mid_mask = cv2.GaussianBlur(mid_mask, (0, 0), sigma)
    high_mask = cv2.GaussianBlur(high_mask, (0, 0), sigma)
    
    # Normalize masks to ensure each frequency point belongs to one band
    total_mask = low_mask + mid_mask + high_mask
    low_mask = low_mask / total_mask
    mid_mask = mid_mask / total_mask
    high_mask = high_mask / total_mask
    
    # Extract frequency components
    low_freq = np.fft.ifftshift(fshift * low_mask)
    mid_freq = np.fft.ifftshift(fshift * mid_mask)
    high_freq = np.fft.ifftshift(fshift * high_mask)
    
    # Inverse Fourier transform
    low_img_y = np.real(np.fft.ifft2(low_freq))
    mid_img_y = np.real(np.fft.ifft2(mid_freq))
    high_img_y = np.real(np.fft.ifft2(high_freq))
    
    # Process color images
    if len(img_float.shape) == 3:
        # Combine Y channel frequency components with original UV channels
        low_img = np.zeros_like(img_float)
        mid_img = np.zeros_like(img_float)
        high_img = np.zeros_like(img_float)
        
        for i in range(3):
            if i == 0:  # Y channel
                low_img[:,:,i] = low_img_y
                mid_img[:,:,i] = mid_img_y
                high_img[:,:,i] = high_img_y
            else:  # UV channels use original values
                low_img[:,:,i] = img_float[:,:,i]
                mid_img[:,:,i] = img_float[:,:,i]
                high_img[:,:,i] = img_float[:,:,i]
        
        # Convert back to RGB
        low_img_rgb = cv2.cvtColor(low_img.astype(np.uint8), cv2.COLOR_YUV2RGB)
        mid_img_rgb = cv2.cvtColor(mid_img.astype(np.uint8), cv2.COLOR_YUV2RGB)
        high_img_rgb = cv2.cvtColor(high_img.astype(np.uint8), cv2.COLOR_YUV2RGB)
        
        low_img = low_img_rgb.astype(np.float32)
        mid_img = mid_img_rgb.astype(np.float32)
        high_img = high_img_rgb.astype(np.float32)
    else:
        # Single channel image
        low_img = low_img_y
        mid_img = mid_img_y
        high_img = high_img_y
    
    # Ensure all components are within valid range
    low_img = np.clip(low_img, 0, 255)
    mid_img = np.clip(mid_img, 0, 255)
    high_img = np.clip(high_img, 0, 255)
    
    return [low_img, mid_img, high_img]
