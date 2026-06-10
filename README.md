# Digital Restoration of Cultural Heritage Sculptures Using LaMa Inpainting

This research project presents an AI-powered framework for the digital restoration of damaged cultural heritage sculptures, with a specific focus on Sursundari idols from historic temples in Maharashtra, India. The objective is to support heritage preservation through automated image reconstruction while maintaining the structural, artistic, and semantic integrity of historical artifacts.

The proposed framework utilizes the LaMa (Large Mask Inpainting) model to restore damaged regions in idol images. Prior to restoration, images undergo preprocessing using computer vision techniques such as noise reduction, contrast enhancement, and mask generation. LaMa's frequency-domain convolution architecture is then employed to reconstruct missing portions while preserving global context, symmetry, and fine-grained sculptural details.

To evaluate restoration quality, the framework incorporates quantitative metrics including Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), Mean Absolute Error (MAE), and restoration percentage. Experimental results achieved an SSIM score of 0.89, PSNR of 32.4 dB, and a restoration rate of 24.6%, demonstrating effective reconstruction with minimal visual artifacts.

The proposed solution provides a scalable, reproducible, and non-invasive approach to digital heritage conservation, supporting archaeological documentation, virtual museum development, and future restoration research.

## Key Features

* AI-powered restoration using LaMa inpainting
* Digital reconstruction of damaged heritage sculptures
* Image preprocessing using OpenCV
* Automated mask generation and damage recovery
* Quantitative evaluation using SSIM, PSNR, and MAE
* Preservation of structural details and texture consistency
* Support for digital archaeology and cultural heritage conservation

## Technologies Used

* Python
* OpenCV
* LaMa (Large Mask Inpainting)
* NumPy
* Computer Vision
* Image Processing
* Deep Learning

## Results

* SSIM: 0.89
* PSNR: 32.4 dB
* Restoration Rate: 24.6%
* High-quality reconstruction with minimal visual artifacts

## Research Objective

The primary objective of this research is to develop an automated and explainable digital restoration framework capable of reconstructing damaged cultural heritage sculptures while preserving their historical and artistic characteristics. The framework aims to assist archaeologists, conservationists, and researchers in heritage documentation, virtual reconstruction, and long-term preservation of cultural artifacts.
