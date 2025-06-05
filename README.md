# JPEG-algorithm

In this project, the JPEG compression algorithm is studied and investigated, and by applying changes to its parameters, different outputs of an image are produced. The aim of this work is to investigate the effect of these changes on the final image quality and the compression rate.

## Overview

JPEG  is a widely used lossy image compression standard that reduces image file size by exploiting spatial redundancy and perceptual limitations of the human visual system. The core steps include color space conversion, chroma subsampling, block splitting, Discrete Cosine Transform (DCT), quantization, and entropy coding (e.g., Huffman coding).

## Project Enhancements and Features

This project is based on the original repository [josgard94/JPEG](https://github.com/josgard94/JPEG) and has been extended and modularized to allow greater flexibility and experimentation:

- Modular design, with distinct scripts for different stages of the JPEG process, for easier modification and testing of different JPEG parameters.
- Configurable parameters such as:
  - Quantization matrices and quality factors
  - Chroma subsampling schemes (e.g., 4:4:4, 4:2:2, 4:2:0)
  - Block sizes for DCT processing
- An automation script (`project_runner.py`) that processes multiple images with various parameter combinations and saves the outputs for comparative analysis.

## Project Structure

The project is structured with the following key Python scripts:

-   `Algoritmo_jpeg.py`: Implements the core JPEG compression steps including color space conversion, chroma subsampling, block splitting, DCT, and quantization.
-   `generador_huffman_code.py`: Generates Huffman codes based on the quantized DCT coefficients.
-   `descompresor.py`: Handles the initial parts of the decompression process.
-   `idtc.py`: Performs the Inverse Discrete Cosine Transform (IDCT) to reconstruct the image.
-   `project_runner.py`: Script to automate the batch processing of multiple images using different parameter sets, calling the above scripts in sequence.
-   `result/`: Folder where compressed images and potentially intermediate files are saved.

## Usage Instructions

1.  Clone or download the repository.
2.  Place input images in a designated input folder (e.g., `input_images/` - you might need to create this or adjust `project_runner.py` accordingly).
3.  Modify parameters for batch processing directly within `project_runner.py` or ensure it's set up to pass them to the other scripts. The individual scripts like `Algoritmo_jpeg.py` might also support interactive parameter input if run standalone.
4.  Run `python project_runner.py` to generate compressed images with different settings. This script will orchestrate the execution of `Algoritmo_jpeg.py`, `generador_huffman_code.py`, `descompresor.py`, and `idtc.py`.
5.  Check the `result/` directory for outputs and compare results.

## Original Repository

This project is derived and extended from the original work by josgard94: [https://github.com/josgard94/JPEG](https://github.com/josgard94/JPEG)

## Acknowledgments

This project is developed for educational and research purposes by modularizing and enhancing the original implementation.

---

If you have any questions or feedback, feel free to open an issue or contact me.
