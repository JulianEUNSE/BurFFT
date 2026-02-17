import numpy as np
import cv2
import os
import shutil
from pathlib import Path

def fourier_transform(image):
    """RGB image Fourier transform, returning amplitude and phase per channel."""
    image = image.astype(np.float32) 

    f = np.fft.fft2(image, axes=(0, 1))
    f_shift = np.fft.fftshift(f, axes=(0, 1))
    amplitude = np.abs(f_shift)
    phase = np.angle(f_shift)
    return amplitude, phase

def inverse_fourier_transform(amplitude, phase):
    """Inverse Fourier transform from amplitude and phase."""
    f_shift = amplitude * np.exp(1j * phase)
    f = np.fft.ifftshift(f_shift, axes=(0, 1))
    image = np.fft.ifft2(f, axes=(0, 1))
    return np.real(image)  

def apply_fda(source_img, target_img, beta=0.01):
    """
    source_img: synthetic image (Unity)
    target_img: real underwater image
    beta: controls the size of frequency region to swap (0.01 to 0.15)
    """

    # Ensure same size
    if source_img.shape != target_img.shape:
        target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))

    h, w, c = source_img.shape

    # Compute Fourier transforms
    source_amp, source_phase = fourier_transform(source_img)
    target_amp, _ = fourier_transform(target_img)

    # Central low-frequency mask
    b = int(min(h, w) * beta)

    mask = np.zeros((h, w), dtype=np.float32)
    center_h, center_w = h // 2, w // 2
    mask[center_h-b:center_h+b, center_w-b:center_w+b] = 1
    mask = np.expand_dims(mask, axis=2) 

    # Swap low-frequency amplitudes
    new_amp = source_amp * (1 - mask) + target_amp * mask

    # Reconstruct image
    adapted_img = inverse_fourier_transform(new_amp, source_phase)

    # Ensure proper range
    adapted_img = np.clip(adapted_img, 0, 255).astype(np.uint8)

    return adapted_img

def process_fda_with_labels(synth_img_dir, synth_label_dir, real_img_dir, output_dir, beta=0.05):
    """
    Process FDA while preserving YOLO labels.
    
    Args:
        synth_img_dir: Directory with synthetic images
        synth_label_dir: Directory with YOLO labels (.txt)
        real_img_dir: Directory with real unlabeled images
        output_dir: Where to save FDA-adapted dataset
        beta: FDA parameter (0.01-0.15)
    """
    # Create output directories
    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get image and label files
    synth_images = sorted([f for f in os.listdir(synth_img_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    real_images = sorted([f for f in os.listdir(real_img_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(synth_images)} synthetic images")
    print(f"Found {len(real_images)} real images")
    
    if len(real_images) == 0:
        raise ValueError("No real images found!")
    
    # Load all real images (for efficiency)
    real_img_data = {}
    for real_file in real_images[:min(100, len(real_images))]:  # Limit to 100 for memory
        img_path = os.path.join(real_img_dir, real_file)
        img = cv2.imread(img_path)
        if img is not None:
            real_img_data[real_file] = img
        else:
            print(f"Warning: Could not load {real_file}")
    
    print(f"Loaded {len(real_img_data)} real images for FDA")
    
    # Process each synthetic image
    for i, synth_file in enumerate(synth_images):
        # Load synthetic image
        synth_path = os.path.join(synth_img_dir, synth_file)
        synth_img = cv2.imread(synth_path)
        
        if synth_img is None:
            print(f"Warning: Could not load {synth_file}, skipping")
            continue
        
        # Get corresponding label file
        label_name = os.path.splitext(synth_file)[0] + '.txt'
        label_path = os.path.join(synth_label_dir, label_name)
        
        # Check if label exists
        if not os.path.exists(label_path):
            print(f"Warning: No label found for {synth_file}, skipping")
            continue
        
        # Select a real image (cycle through available ones)
        real_key = list(real_img_data.keys())[i % len(real_img_data)]
        real_img = real_img_data[real_key]
        
        # Apply FDA
        try:
            adapted_img = apply_fda(synth_img, real_img, beta=beta)
            
            # Save FDA-adapted image
            output_img_path = os.path.join(output_img_dir, synth_file)
            cv2.imwrite(output_img_path, adapted_img)
            
            # Copy corresponding label (unchanged)
            output_label_path = os.path.join(output_label_dir, label_name)
            shutil.copy2(label_path, output_label_path)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(synth_images)} images")
                
        except Exception as e:
            print(f"Error processing {synth_file}: {e}")
    
    print(f"\n FDA Complete")
    print(f"  Adapted images: {output_img_dir}")
    print(f"  Labels: {output_label_dir}")
    print(f"  Total processed: {len(synth_images)} images")
    
    return output_img_dir, output_label_dir

def create_yolo_dataset_yaml(output_dir, class_names):
    """
    Create YOLO dataset.yaml file for training.
    
    Args:
        output_dir: Directory with FDA-adapted data
        class_names: List of class names (e.g., ['gate', 'bin', 'path_marker'])
    """
    yaml_content = f"""# YOLO dataset configuration for FDA-adapted data
path: {os.path.abspath(output_dir)}  # dataset root dir
train: images  # train images (relative to 'path')
val: images    # using same for validation (or split manually)

# Class names
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_file = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YOLO config: {yaml_file}")
    return yaml_file

# Main execution
if __name__ == "__main__":
    # ===== CONFIGURE THESE PATHS =====
    SYNTH_IMG_DIR = "synthetic/images"
    SYNTH_LABEL_DIR = "synthetic/labels"
    REAL_IMG_DIR = "real/images"
    OUTPUT_DIR = "fda_adapted"  # Will create this folder
    BETA = 0.01  # adjust based on results
    
    # Your object classes (CHANGE THESE!)
    CLASS_NAMES = ["Gate", "Bin", "Path", "White Slalom", "Red Slalom", "Map"]  # Update with your actual classes

    # ===== RUN FDA PROCESSING =====
    print("Starting FDA processing...")
    
    img_dir, label_dir = process_fda_with_labels(
        synth_img_dir=SYNTH_IMG_DIR,
        synth_label_dir=SYNTH_LABEL_DIR,
        real_img_dir=REAL_IMG_DIR,
        output_dir=OUTPUT_DIR,
        beta=BETA
    )
    
    yaml_path = create_yolo_dataset_yaml(OUTPUT_DIR, CLASS_NAMES)
    
