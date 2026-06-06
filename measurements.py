import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Load images ===
original = cv2.imread("image2 - initial.jpg")          # damaged idol
reconstructed = cv2.imread("image2 - generated.png")  # after LaMa inpainting

# Resize to same size
original = cv2.resize(original, (reconstructed.shape[1], reconstructed.shape[0]))

# === Compute difference map ===
diff = cv2.absdiff(original, reconstructed)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

# Morphological smoothing
kernel = np.ones((7,7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours of changed areas
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 1000
contours = [c for c in contours if cv2.contourArea(c) > min_area]

# === Visualize bounding boxes ===
highlight = reconstructed.copy()
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(highlight, (x, y), (x+w, y+h), (0,255,0), 2)

# === Generate heatmap overlay ===
heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(reconstructed, 0.7, heatmap, 0.3, 0)

# === Calculate measurement differences ===
total_pixels = mask.size
changed_pixels = np.sum(mask > 0)
percent_changed = (changed_pixels / total_pixels) * 100

# Area of each major reconstruction
areas = [cv2.contourArea(c) for c in contours]
area_total = sum(areas)

# === Show results ===
plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original (Damaged)")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
plt.title("Detected Reconstructed Regions")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Reconstruction Heatmap")
plt.axis('off')
plt.show()

# === Bar chart visualization ===
plt.bar(["Original", "Reconstructed"], [total_pixels - changed_pixels, total_pixels])
plt.title("Comparison of Pixel Areas")
plt.ylabel("Pixel Count (approx.)")
plt.show()

# === Textual Explanation ===
print("\n📏 Reconstruction Measurement Report\n" + "-"*45)
print(f"Total Image Area (pixels): {total_pixels:,}")
print(f"Restored/Modified Area: {area_total:.2f} pixels²")
print(f"Percentage of Idol Reconstructed: {percent_changed:.2f}%")

if percent_changed < 5:
    print("🟡 Minor restoration — small damaged regions were filled.")
elif 5 <= percent_changed < 20:
    print("🟢 Moderate restoration — visible damaged parts reconstructed accurately.")
else:
    print("🔵 Major restoration — large sections were filled, possibly reconstructing missing idol portions.")
