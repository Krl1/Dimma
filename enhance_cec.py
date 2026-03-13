import os
import csv
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from src.models import LitDimma
from src.transforms.pair_transform import PairedTransformForDimma

def main():
    # 1. Setup paths and config
    config_path = "configs/CEC/stage2/400shot-cec-ft.yaml"
    checkpoint_path = "checkpoints/dimma-400shot-cec-ft/dimma-400shot-cec-ft.ckpt"
    test_csv_path = "../../datasets/cec_dataset_only_dim/test.csv"
    output_dir = "../../results/dimma"
    images_output_dir = os.path.join(output_dir, "images")
    results_csv_path = os.path.join(output_dir, "results.csv")

    os.makedirs(images_output_dir, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model
    print(f"Loading model from {checkpoint_path}...")
    model = LitDimma.load_from_checkpoint(checkpoint_path, config=cfg, weights_only=False)
    model.to(device)
    model.eval()

    # 3. Setup transforms
    # Using PairedTransformForDimma(test=True) as in the original pipeline
    transform = PairedTransformForDimma(crop_size=416, test=True)

    # 4. Read test CSV
    results = []
    with open(test_csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Enhancing {len(rows)} images...")
    
    with torch.no_grad():
        for row in tqdm(rows):
            corrupted_path = row['corrupted']
            clean_path = row['clean']
            
            if not os.path.exists(corrupted_path):
                # Try relative path if absolute fails (sometimes datasets are moved)
                # But here they seem absolute to /home/kkulawiec/...
                pass

            # Load images
            corrupted_img = cv2.imread(corrupted_path)[..., ::-1]
            clean_img = cv2.imread(clean_path)[..., ::-1]

            # Transform
            transformed = transform(image=corrupted_img, target=clean_img)
            
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            source_lightness = transformed['source_lightness'].unsqueeze(0).to(device)
            target_lightness = transformed['target_lightness'].unsqueeze(0).to(device)

            # Inference
            # forward(self, image, source_lightness, target_lightness)
            enhanced_tensor = model(input_tensor, source_lightness, target_lightness)
            
            # Post-process
            enhanced_numpy = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            enhanced_numpy = (enhanced_numpy * 255).astype(np.uint8)
            
            # Resize back to 420x420 as requested
            enhanced_numpy = cv2.resize(enhanced_numpy, (420, 420), interpolation=cv2.INTER_LANCZOS4)
            
            enhanced_bgr = enhanced_numpy[..., ::-1]

            # Save image
            image_name = os.path.basename(corrupted_path)
            processed_path = os.path.abspath(os.path.join(images_output_dir, image_name))
            cv2.imwrite(processed_path, enhanced_bgr)

            # Store result info
            results.append({
                'source': os.path.abspath(corrupted_path),
                'processed': processed_path,
                'target': os.path.abspath(clean_path)
            })

    # 5. Save results CSV
    with open(results_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'processed', 'target'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Enhanced images saved in {images_output_dir}")
    print(f"Results summary saved in {results_csv_path}")

if __name__ == "__main__":
    main()
