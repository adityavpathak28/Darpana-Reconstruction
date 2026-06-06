import torch
import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# Paths (use Path objects to avoid Windows path errors)
sam_checkpoint = PROJECT_DIR / "models" / "sam_vit_h_4b8939.pth"
controlnet_model_path = PROJECT_DIR / "models" / "controlnet-canny"
stable_diffusion_model_path = PROJECT_DIR / "models" / "stable-diffusion-v1-5-diffusers"

input_image_path = PROJECT_DIR / "image1 - initial.jpg"
output_dir = PROJECT_DIR / "output"
output_dir.mkdir(exist_ok=True)

device = "cpu"

# -----------------------------
# STEP 1: SEGMENT IMAGE WITH SAM
# -----------------------------
print("[1/4] Loading SAM model...")
sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
mask_generator = SamAutomaticMaskGenerator(sam)

print("[2/4] Generating segmentation mask...")
image_bgr = cv2.imread(str(input_image_path))
if image_bgr is None:
    raise FileNotFoundError(f"❌ Could not find input image at {input_image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image_rgb)

mask_image = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
for mask in masks:
    mask_image[mask["segmentation"]] = 255

mask_output_path = output_dir / "mask_from_sam.png"
cv2.imwrite(str(mask_output_path), mask_image)
print(f"✅ SAM mask saved at {mask_output_path}")

# -----------------------------
# STEP 2: GENERATE CANNY EDGES
# -----------------------------
print("[3/4] Extracting Canny edges...")
edges = cv2.Canny(image_bgr, 100, 200)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

canny_output_path = output_dir / "canny_edges.png"
cv2.imwrite(str(canny_output_path), edges)
print(f"✅ Canny edges saved at {canny_output_path}")

# -----------------------------
# STEP 3: LOAD CONTROLNET + STABLE DIFFUSION
# -----------------------------
print("[4/4] Loading ControlNet + Stable Diffusion pipeline (CPU mode)...")

controlnet = ControlNetModel.from_pretrained(
    str(controlnet_model_path),
    torch_dtype=torch.float32
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path=str(stable_diffusion_model_path.as_posix()),
    controlnet=controlnet,
    torch_dtype=torch.float32
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# -----------------------------
# STEP 4: RECONSTRUCTION
# -----------------------------
prompt = (
    "Restored and complete ancient idol sculpture, realistic texture, detailed surface, "
    "artifact restoration, high-quality museum photo"
)
negative_prompt = "blurry, distorted, deformed, unrealistic, artifacts, broken, noisy"

control_image = Image.open(canny_output_path).convert("RGB")

print("🚀 Running Stable Diffusion with ControlNet (CPU mode)... This may take several minutes...")
with torch.inference_mode():
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        generator=torch.manual_seed(42),
        image=control_image
    ).images[0]

output_path = output_dir / "idol_reconstructed.png"
output.save(str(output_path))
print(f"✅ Idol reconstruction completed successfully!\nOutput saved at: {output_path}")
