import numpy as np
import scipy.fftpack
from PIL import Image, UnidentifiedImageError
import os
import argparse
import sys

def load_numpy_array_from_file(filename):
    try:
        arr = np.loadtxt(filename, dtype=int)
        if arr.ndim == 0 and os.path.getsize(filename) == 0: return np.array([[]], dtype=int)
        return arr
    except Exception as e:
        print(f"    [!] ERROR (load_numpy): Could not load array from '{filename}': {e}")
        return None

def inverse_dct_block(block_coeffs_NxN, quant_matrix_channel_NxN):
    dequantized = block_coeffs_NxN * quant_matrix_channel_NxN
    idct_block_transformed = scipy.fftpack.idct(
        scipy.fftpack.idct(dequantized, axis=0, norm='ortho'),
        axis=1, norm='ortho'
    )
    return idct_block_transformed + 128

def reconstruct_channel(flat_coeffs_channel, channel_height, channel_width, quant_matrix_channel_NxN, block_size_N):
    if channel_height == 0 or channel_width == 0 or block_size_N == 0:
        return np.zeros((channel_height, channel_width), dtype=np.uint8)
    expected_coeffs_in_channel = channel_height * channel_width
    if len(flat_coeffs_channel) != expected_coeffs_in_channel:
        print(f"    [!] ERROR (reconstruct_channel): Coeff count {len(flat_coeffs_channel)} != expected {expected_coeffs_in_channel} for {channel_width}x{channel_height}")
        raise ValueError("Coefficient count mismatch in reconstruct_channel.")

    coeffs_matrix_2d_NxN_blocks = flat_coeffs_channel.reshape((channel_height, channel_width))
    reconstructed_channel_array = np.zeros((channel_height, channel_width))

    for r_idx in range(0, channel_height, block_size_N):
        for c_idx in range(0, channel_width, block_size_N):
            if r_idx + block_size_N <= channel_height and c_idx + block_size_N <= channel_width:
                block_coeffs_data = coeffs_matrix_2d_NxN_blocks[r_idx:r_idx+block_size_N, c_idx:c_idx+block_size_N]
                if block_coeffs_data.shape == (block_size_N, block_size_N):
                    spatial_block_data = inverse_dct_block(block_coeffs_data, quant_matrix_channel_NxN)
                    reconstructed_channel_array[r_idx:r_idx+block_size_N, c_idx:c_idx+block_size_N] = spatial_block_data
    return np.clip(reconstructed_channel_array, 0, 255).astype(np.uint8)

def run_idtc(input_matrix_file, input_dims_file, input_q_y_file, input_q_c_file, output_image_file_name):
    if not all(os.path.exists(f) for f in [input_matrix_file, input_dims_file, input_q_y_file, input_q_c_file]):
        print(f"    [!] ERROR (run_idtc): One or more input files not found for IDTC."); return False

    try:
        with open(input_dims_file, "r", encoding='utf-8') as dim_file:
            y_width_proc = int(dim_file.readline().strip())
            y_height_proc = int(dim_file.readline().strip())
            cb_width_proc = int(dim_file.readline().strip())
            cb_height_proc = int(dim_file.readline().strip())
            cr_width_proc = int(dim_file.readline().strip())
            cr_height_proc = int(dim_file.readline().strip())
            subsampling_mode_str = dim_file.readline().strip()
            block_size_N_read = int(dim_file.readline().strip())
    except Exception as e:
        print(f"    [!] ERROR (run_idtc) reading dims file '{input_dims_file}': {e}"); return False

    quant_matrix_y_used = load_numpy_array_from_file(input_q_y_file)
    quant_matrix_c_used = load_numpy_array_from_file(input_q_c_file)
    if quant_matrix_y_used is None or quant_matrix_y_used.size == 0 or \
       quant_matrix_c_used is None or quant_matrix_c_used.size == 0 :
        print(f"    [!] ERROR (run_idtc): Quantization matrix files could not be loaded or are empty."); return False

    if quant_matrix_y_used.shape != (block_size_N_read, block_size_N_read) or \
       quant_matrix_c_used.shape != (block_size_N_read, block_size_N_read):
        print(f"    [!] ERROR (run_idtc): Q matrix dimensions mismatch block size {block_size_N_read}x{block_size_N_read}.")
        return False

    try:
        with open(input_matrix_file, 'r', encoding='utf-8') as f: all_coeffs_str = f.read().strip().split()
    except Exception as e:
        print(f"    [!] ERROR (run_idtc) reading matrix file '{input_matrix_file}': {e}"); return False

    if not all_coeffs_str or (len(all_coeffs_str) == 1 and not all_coeffs_str[0]):
        print(f"    [!] WARNING (run_idtc): Matrix file '{input_matrix_file}' is empty. Creating black/dummy image.");
        if y_width_proc > 0 and y_height_proc > 0:
            dummy_image = Image.new('RGB', (y_width_proc, y_height_proc), color='black')
            try: dummy_image.save(output_image_file_name)
            except Exception as e_save: print(f"    [!] ERROR (run_idtc) saving blank image: {e_save}")
        else:
             try: Image.new('RGB', (1,1), color='black').save(output_image_file_name)
             except: pass
        return True

    try:
        all_coeffs_flat = np.array([float(s) for s in all_coeffs_str])
    except ValueError as e:
        print(f"    [!] ERROR (run_idtc) converting coeffs in '{input_matrix_file}': {e}"); return False

    num_coeffs_y = y_width_proc * y_height_proc if y_width_proc > 0 and y_height_proc > 0 else 0
    num_coeffs_cb = cb_width_proc * cb_height_proc if cb_width_proc > 0 and cb_height_proc > 0 else 0
    num_coeffs_cr = cr_width_proc * cr_height_proc if cr_width_proc > 0 and cr_height_proc > 0 else 0
    expected_total_coeffs_from_dims = num_coeffs_y + num_coeffs_cb + num_coeffs_cr

    if len(all_coeffs_flat) != expected_total_coeffs_from_dims:
        print(f"    [!] ERROR (run_idtc): Total coeffs {len(all_coeffs_flat)} != expected {expected_total_coeffs_from_dims}.")
        return False

    coeffs_Y_flat = all_coeffs_flat[0 : num_coeffs_y] if num_coeffs_y > 0 else np.array([])
    coeffs_Cb_flat = all_coeffs_flat[num_coeffs_y : num_coeffs_y + num_coeffs_cb] if num_coeffs_cb > 0 else np.array([])
    coeffs_Cr_flat = all_coeffs_flat[num_coeffs_y + num_coeffs_cb : expected_total_coeffs_from_dims] if num_coeffs_cr > 0 else np.array([])
    
    try:
        Y_recon_array = reconstruct_channel(coeffs_Y_flat, y_height_proc, y_width_proc, quant_matrix_y_used, block_size_N_read)
        Cb_recon_array_small = reconstruct_channel(coeffs_Cb_flat, cb_height_proc, cb_width_proc, quant_matrix_c_used, block_size_N_read)
        Cr_recon_array_small = reconstruct_channel(coeffs_Cr_flat, cr_height_proc, cr_width_proc, quant_matrix_c_used, block_size_N_read)
    except ValueError as e:
        print(f"    [!] ERROR (run_idtc) during channel reconstruction: {e}"); return False

    if y_width_proc == 0 or y_height_proc == 0:
        print("    [!] ERROR (run_idtc): Y channel has zero dimensions, cannot create image.")
        try: Image.new('RGB', (1,1), color='black').save(output_image_file_name)
        except: pass
        return False

    Y_img = Image.fromarray(Y_recon_array, mode='L') if Y_recon_array.size > 0 else Image.new('L', (y_width_proc, y_height_proc) if y_width_proc > 0 else (1,1))
    Cb_img_small = Image.fromarray(Cb_recon_array_small, mode='L') if Cb_recon_array_small.size > 0 else Image.new('L', (0,0))
    Cr_img_small = Image.fromarray(Cr_recon_array_small, mode='L') if Cr_recon_array_small.size > 0 else Image.new('L', (0,0))

    resample_filter = Image.Resampling.LANCZOS
    final_y_width, final_y_height = y_width_proc, y_height_proc

    if subsampling_mode_str == "420" or subsampling_mode_str == "422":
        Cb_img_upsampled = Cb_img_small.resize((final_y_width, final_y_height), resample_filter) if Cb_img_small.width > 0 and Cb_img_small.height > 0 else Image.new('L', (final_y_width, final_y_height), color=128)
        Cr_img_upsampled = Cr_img_small.resize((final_y_width, final_y_height), resample_filter) if Cr_img_small.width > 0 and Cr_img_small.height > 0 else Image.new('L', (final_y_width, final_y_height), color=128)
    else:  # 444
        Cb_img_upsampled = Cb_img_small
        Cr_img_upsampled = Cr_img_small
        if Cb_img_upsampled.width > 0 and Cb_img_upsampled.size != Y_img.size : Cb_img_upsampled = Cb_img_upsampled.resize(Y_img.size, resample_filter)
        elif Cb_img_upsampled.width == 0 and Y_img.size > 0 : Cb_img_upsampled = Image.new('L', Y_img.size, color=128)
        if Cr_img_upsampled.width > 0 and Cr_img_upsampled.size != Y_img.size : Cr_img_upsampled = Cr_img_upsampled.resize(Y_img.size, resample_filter)
        elif Cr_img_upsampled.width == 0 and Y_img.size > 0 : Cr_img_upsampled = Image.new('L', Y_img.size, color=128)
    
    if Y_img.width == 0 or Y_img.height == 0 :
        print(f"    [!] ERROR (run_idtc): Y channel image is empty after reconstruction. Cannot merge."); return False

    try:
        ycbcr_reconstructed_image = Image.merge('YCbCr', (Y_img, Cb_img_upsampled, Cr_img_upsampled))
        rgb_reconstructed_image = ycbcr_reconstructed_image.convert('RGB')
    except Exception as e_merge:
        print(f"    [!] ERROR (run_idtc) merging channels or converting to RGB: {e_merge}"); return False

    try:
        rgb_reconstructed_image.save(output_image_file_name)
        print(f"    [+] INFO (run_idtc): Decompressed image saved as '{os.path.basename(output_image_file_name)}'.")
    except Exception as e:
        print(f"    [!] ERROR (run_idtc) saving final image '{output_image_file_name}': {e}"); return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructs an image from DCT coefficients.")
    parser.add_argument("--matrix_file", default="matrix.txt", help="Path to the input DCT coefficients matrix file.")
    parser.add_argument("--dims_file", default="dims.txt", help="Path to the dimensions and settings file.")
    parser.add_argument("--q_y_file", default="q_matrix_y_used.txt", help="Path to the Y quantization matrix file.")
    parser.add_argument("--q_c_file", default="q_matrix_c_used.txt", help="Path to the C quantization matrix file.")
    parser.add_argument("--output_image", default="decompressed_image.jpg", help="Path for the output reconstructed image.")
    args = parser.parse_args()

    if not run_idtc(args.matrix_file, args.dims_file, args.q_y_file, args.q_c_file, args.output_image):
        print("❌ Image reconstruction failed when run independently.")
        sys.exit(1)
    # else:
        # print(f"✅ Image reconstruction successful. Output: {args.output_image} (when run independently).")