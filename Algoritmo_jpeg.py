from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import argparse

# --- Quantization Matrices ---
standard_jpeg_q_y_8x8 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=int)

standard_jpeg_q_c_8x8 = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=int)

def generate_base_quant_matrix_hvs(block_size, base_factor=1.0, is_chroma=False):
    matrix = np.zeros((block_size, block_size), dtype=np.float32)
    if block_size == 8: dc_val = 16 if not is_chroma else 17
    elif block_size == 4: dc_val = 8 if not is_chroma else 9
    elif block_size == 16: dc_val = 32 if not is_chroma else 34
    else: dc_val = 16 if not is_chroma else 17
    matrix[0, 0] = dc_val
    for i in range(block_size):
        for j in range(block_size):
            if i == 0 and j == 0: continue
            normalized_freq = np.sqrt((i/block_size)**2 + (j/block_size)**2)
            val = dc_val + base_factor * normalized_freq * (50 + block_size * 2)
            matrix[i, j] = val
    return matrix

def normalize_quant_matrix(matrix_input, max_val=255.0, min_val=1.0):
    matrix = matrix_input.copy().astype(np.float32)
    dc_val = matrix[0,0]
    if matrix.size > 1:
        ac_mask = np.ones(matrix.shape, dtype=bool); ac_mask[0, 0] = False
        ac_elements = matrix[ac_mask]
        if ac_elements.size > 0:
            min_ac = ac_elements.min(); max_ac = ac_elements.max()
            if max_ac > min_ac:
                matrix[ac_mask] = min_val + (ac_elements - min_ac) * (max_val - min_val) / (max_ac - min_ac)
            elif ac_elements.size > 0 : matrix[ac_mask] = min_val
    matrix[0,0] = np.clip(dc_val, min_val, max_val)
    matrix = np.clip(matrix, min_val, max_val)
    return matrix.astype(np.float32)

def scale_quant_matrix(input_matrix_NxN_float, quality_level):
    if not (1 <= quality_level <= 100): quality_level = np.clip(quality_level, 1, 100)
    if quality_level == 50: multiplier = 1.0
    elif quality_level < 50: multiplier = 1.0 + ((50.0 - quality_level) / 49.0) * 9.0
    else: multiplier = 1.0 - ((quality_level - 50.0) / 50.0) * 0.95
    if multiplier < 0.01: multiplier = 0.01
    scaled_matrix = input_matrix_NxN_float * multiplier
    scaled_matrix = np.round(scaled_matrix)
    scaled_matrix[scaled_matrix < 1] = 1
    scaled_matrix[scaled_matrix > 255] = 255
    return scaled_matrix.astype(int)

temporal_store = {}
frequency_table = {}
global_index = 0

def dct_2d(block_NxN, quant_matrix_channel_NxN):
    current_block_size = block_NxN.shape[0]
    shifted_block = block_NxN - 128
    dct_block_transformed = dct(dct(shifted_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    quantized_block_NxN = np.zeros((current_block_size, current_block_size))
    for i in range(current_block_size):
        for j in range(current_block_size):
            if quant_matrix_channel_NxN[i][j] == 0: quantized_block_NxN[i, j] = dct_block_transformed[i, j]
            else: quantized_block_NxN[i, j] = np.fix(dct_block_transformed[i, j] / quant_matrix_channel_NxN[i, j])
    return quantized_block_NxN

def zigzag_ordering(block_NxN):
    global global_index, temporal_store, frequency_table
    N_current = block_NxN.shape[0]
    if N_current == 0: return
    linearized_coeffs = np.empty(N_current * N_current, dtype=block_NxN.dtype)
    idx = 0
    for s_val in range(2 * N_current - 1):
        if s_val % 2 == 0:
            r, c = (s_val, 0) if s_val < N_current else (N_current - 1, s_val - N_current + 1)
            while r >= 0 and c < N_current:
                if idx < linearized_coeffs.size: linearized_coeffs[idx] = block_NxN[r, c]; idx += 1
                else: break
                r -= 1; c += 1
        else:
            c, r = (s_val, 0) if s_val < N_current else (N_current - 1, s_val - N_current + 1)
            while c >= 0 and r < N_current:
                if idx < linearized_coeffs.size: linearized_coeffs[idx] = block_NxN[r, c]; idx += 1
                else: break
                c -= 1; r += 1
    if idx != N_current * N_current:
        # print(f"WARNING (zigzag): Array length ({idx}) != {N_current*N_current}. Padding with zeros.")
        linearized_coeffs[idx:] = 0

    for i in range(N_current * N_current):
        val_coeff = linearized_coeffs[i]
        temporal_store[global_index] = val_coeff
        if val_coeff not in frequency_table: frequency_table[val_coeff] = 0
        frequency_table[val_coeff] += 1
        global_index += 1

def save_probabilities(prob_dict_final, keys_final, filepath):
    with open(filepath, "w", encoding='utf-8') as file:
        for key_val in keys_final:
            probability_value = prob_dict_final.get(key_val, 0.0)
            file.write(f"{str(key_val)}\t{probability_value}\n")

def save_numpy_array_to_file(array, filename):
    np.savetxt(filename, array, fmt='%d')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG-like image compression.")
    parser.add_argument("--input_image", required=True, help="Path to the input image file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save intermediate and output files.")
    parser.add_argument("--quality", type=int, default=75, help="Quality level (1-100).")
    parser.add_argument("--subsampling", type=str, default="4:2:0", choices=["4:4:4", "4:2:2", "4:2:0"], help="Chroma subsampling mode.")
    parser.add_argument("--block_size", type=int, default=8, choices=[4, 8, 16], help="DCT block size.")
    parser.add_argument("--standard_mode", action="store_true", help="Use standard JPEG settings (Q50, 4:2:0, 8x8 block).")
    args = parser.parse_args()

    temporal_store = {}
    frequency_table = {}
    global_index = 0

    os.makedirs(args.output_dir, exist_ok=True)

    dct_file_path = os.path.join(args.output_dir, "dct.txt")
    result_file_path = os.path.join(args.output_dir, "result.txt")
    dims_file_path = os.path.join(args.output_dir, "dims.txt")
    q_matrix_y_file = os.path.join(args.output_dir, "q_matrix_y_used.txt")
    q_matrix_c_file = os.path.join(args.output_dir, "q_matrix_c_used.txt")
    y_channel_processed_file = os.path.join(args.output_dir, "Y_channel_processed.jpg")
    dct_y_visual_file = os.path.join(args.output_dir, "dct_Y_channel_visual.jpg")

    if args.standard_mode:
        quality_level = 50
        subsampling_mode_input = "4:2:0"
        block_size_N = 8
        use_standard_q_matrices_flag = True
        print(f"[+] INFO: Running in Standard Mode for {os.path.basename(args.input_image)}") # ایموجی حذف شد
    else:
        quality_level = args.quality
        subsampling_mode_input = args.subsampling
        block_size_N = args.block_size
        use_standard_q_matrices_flag = (block_size_N == 8 and quality_level == 50)
        print(f"[*] INFO: Running in Dynamic Mode for {os.path.basename(args.input_image)} (Q={quality_level}, Sub={subsampling_mode_input}, Block={block_size_N})") # ایموجی حذف شد

    try:
        image = Image.open(args.input_image)
        if image.mode == 'CMYK': image = image.convert('RGB')
    except FileNotFoundError:
        print(f"[!] FATAL: Input image '{args.input_image}' not found. Exiting.")
        exit(1)
    except UnidentifiedImageError:
        print(f"[!] FATAL: Cannot identify image file '{args.input_image}'. May be corrupted or unsupported. Exiting.")
        exit(1)
    except Exception as e:
        print(f"[!] FATAL: Error opening image '{args.input_image}': {e}. Exiting."); exit(1)
    
    print(f"    Image '{os.path.basename(args.input_image)}' loaded. Mode: {image.mode}, Size: {image.size}")

    if use_standard_q_matrices_flag:
        current_quant_matrix_y = standard_jpeg_q_y_8x8.copy()
        current_quant_matrix_c = standard_jpeg_q_c_8x8.copy()
        print("    INFO: Using standard JPEG quantization matrices (8x8, Quality 50).")
    else:
        print(f"    INFO: Generating dynamic quantization matrices for block {block_size_N}x{block_size_N}, quality {quality_level}.")
        base_y_NxN_hvs = generate_base_quant_matrix_hvs(block_size_N, base_factor=1.0, is_chroma=False)
        base_c_NxN_hvs = generate_base_quant_matrix_hvs(block_size_N, base_factor=1.2, is_chroma=True)
        normalized_y_NxN = normalize_quant_matrix(base_y_NxN_hvs, max_val=255.0, min_val=1.0)
        normalized_c_NxN = normalize_quant_matrix(base_c_NxN_hvs, max_val=255.0, min_val=1.0)
        current_quant_matrix_y = scale_quant_matrix(normalized_y_NxN, quality_level)
        current_quant_matrix_c = scale_quant_matrix(normalized_c_NxN, quality_level)

    save_numpy_array_to_file(current_quant_matrix_y, q_matrix_y_file)
    save_numpy_array_to_file(current_quant_matrix_c, q_matrix_c_file)
    # print("    INFO: Quantization matrices set and saved.")

    ycbcr_image = image.convert('YCbCr')
    Y_pil, Cb_pil, Cr_pil = ycbcr_image.split()
    original_width, original_height = image.size

    y_width_base = (original_width // block_size_N) * block_size_N
    y_height_base = (original_height // block_size_N) * block_size_N
    if y_width_base == 0 or y_height_base == 0:
        print(f"[!] FATAL: Image dimensions too small for block size {block_size_N}. Y_proc: {y_width_base}x{y_height_base}. Exiting."); exit(1)
    Y_pil_proc = Y_pil.crop((0, 0, y_width_base, y_height_base))

    resample_filter = Image.Resampling.LANCZOS
    if subsampling_mode_input == "4:2:2":
        cb_width_sub_target = y_width_base // 2; cb_height_sub_target = y_height_base
        cr_width_sub_target = y_width_base // 2; cr_height_sub_target = y_height_base
    elif subsampling_mode_input == "4:2:0":
        cb_width_sub_target = y_width_base // 2; cb_height_sub_target = y_height_base // 2
        cr_width_sub_target = y_width_base // 2; cr_height_sub_target = y_height_base // 2
    else:  # 4:4:4
        cb_width_sub_target = y_width_base; cb_height_sub_target = y_height_base
        cr_width_sub_target = y_width_base; cr_height_sub_target = y_height_base

    Cb_pil_sub = Cb_pil.resize((cb_width_sub_target, cb_height_sub_target), resample_filter) if cb_width_sub_target > 0 and cb_height_sub_target > 0 else Image.new('L', (0,0))
    Cr_pil_sub = Cr_pil.resize((cr_width_sub_target, cr_height_sub_target), resample_filter) if cr_width_sub_target > 0 and cr_height_sub_target > 0 else Image.new('L', (0,0))

    cb_width_proc = (Cb_pil_sub.width // block_size_N) * block_size_N
    cb_height_proc = (Cb_pil_sub.height // block_size_N) * block_size_N
    cr_width_proc = (Cr_pil_sub.width // block_size_N) * block_size_N
    cr_height_proc = (Cr_pil_sub.height // block_size_N) * block_size_N

    Cb_pil_proc = Cb_pil_sub.crop((0,0, cb_width_proc, cb_height_proc)) if cb_width_proc > 0 and cb_height_proc > 0 else Image.new('L', (0,0))
    Cr_pil_proc = Cr_pil_sub.crop((0,0, cr_width_proc, cr_height_proc)) if cr_width_proc > 0 and cr_height_proc > 0 else Image.new('L', (0,0))

    # print(f"    INFO: Y processed dims: {y_width_base}x{y_height_base}")
    # print(f"    INFO: Cb processed dims: {cb_width_proc}x{cb_height_proc}")
    # print(f"    INFO: Cr processed dims: {cr_width_proc}x{cr_height_proc}")

    channels_data_map = {
        'Y': {'pil': Y_pil_proc, 'q_matrix': current_quant_matrix_y, 'width': y_width_base, 'height': y_height_base},
        'Cb': {'pil': Cb_pil_proc, 'q_matrix': current_quant_matrix_c, 'width': cb_width_proc, 'height': cb_height_proc},
        'Cr': {'pil': Cr_pil_proc, 'q_matrix': current_quant_matrix_c, 'width': cr_width_proc, 'height': cr_height_proc}
    }
    all_dct_coeffs_linear_list = []

    for ch_name, data in channels_data_map.items():
        ch_pil_image, ch_q_matrix, ch_width, ch_height = data['pil'], data['q_matrix'], data['width'], data['height']
        if ch_width == 0 or ch_height == 0: continue
        # print(f"    INFO: Processing channel {ch_name} ({ch_width}x{ch_height}, block: {block_size_N}x{block_size_N})...")

        image_array_ch = np.asarray(ch_pil_image, dtype=np.float32)
        dct_ch_matrix = np.zeros((ch_height, ch_width))

        for r_idx in range(0, ch_height, block_size_N):
            for c_idx in range(0, ch_width, block_size_N):
                if r_idx + block_size_N <= ch_height and c_idx + block_size_N <= ch_width:
                    block_NxN_data = image_array_ch[r_idx:r_idx+block_size_N, c_idx:c_idx+block_size_N]
                    if block_NxN_data.shape == (block_size_N, block_size_N):
                        quant_dct_block = dct_2d(block_NxN_data, ch_q_matrix)
                        quant_dct_block[quant_dct_block == -0.0] = 0.0
                        dct_ch_matrix[r_idx:r_idx+block_size_N, c_idx:c_idx+block_size_N] = quant_dct_block
                        zigzag_ordering(quant_dct_block)
        all_dct_coeffs_linear_list.extend(dct_ch_matrix.flatten())

        if ch_name == 'Y' and y_channel_processed_file and dct_y_visual_file:
            try:
                if image_array_ch.size > 0: Image.fromarray(image_array_ch.astype(np.uint8)).save(y_channel_processed_file)
                if dct_ch_matrix.size > 0 :
                    vis_dct_y = np.log(np.abs(dct_ch_matrix) + 1)
                    if np.max(vis_dct_y) > 0: vis_dct_y_norm = (vis_dct_y / np.max(vis_dct_y) * 255).astype(np.uint8)
                    else: vis_dct_y_norm = vis_dct_y.astype(np.uint8)
                    Image.fromarray(vis_dct_y_norm).save(dct_y_visual_file)
            except Exception as e_img_save: print(f"    [!] WARNING: Could not save intermediate Y channel images: {e_img_save}")

    with open(dct_file_path, "w", encoding='utf-8') as f: f.write(" ".join(map(str, all_dct_coeffs_linear_list)))
    with open(dims_file_path, "w", encoding='utf-8') as dim_file:
        dim_file.write(f"{y_width_base}\n{y_height_base}\n")
        dim_file.write(f"{cb_width_proc}\n{cb_height_proc}\n")
        dim_file.write(f"{cr_width_proc}\n{cr_height_proc}\n")
        dim_file.write(f"{subsampling_mode_input.replace(':', '')}\n")
        dim_file.write(f"{block_size_N}\n")
    # print(f"    INFO: Intermediate files saved in '{args.output_dir}'.")

    unique_values_from_table = list(frequency_table.keys())
    total_coeffs_counted = sum(frequency_table.values())

    if total_coeffs_counted > 0:
        probabilities_final_dict = {val: count / float(total_coeffs_counted) for val, count in frequency_table.items()}
        save_probabilities(probabilities_final_dict, unique_values_from_table, result_file_path)
    else:
        print("    [!] WARNING: No coefficients for probability calculation. Creating empty result.txt.")
        with open(result_file_path, "w", encoding='utf-8') as file: file.write("0.0\t0.0\n")

    print(f"    [+] INFO: DCT and Quantization for '{os.path.basename(args.input_image)}' completed.")
    # expected_total_coeffs = (y_width_base * y_height_base) + \
    #                         (cb_width_proc * cb_height_proc if cb_width_proc > 0 and cb_height_proc > 0 else 0) + \
    #                         (cr_width_proc * cr_height_proc if cr_width_proc > 0 and cr_height_proc > 0 else 0)
    # print(f"    INFO: Coeffs written (dct.txt): {len(all_dct_coeffs_linear_list)}, Expected (processed dims): {expected_total_coeffs}")
    # if len(all_dct_coeffs_linear_list) != expected_total_coeffs:
    #     print(f"    [!] WARNING: Mismatch in written coefficients vs expected!")
    print(f"    [i] INFO: Settings - Subsampling: {subsampling_mode_input}, Block: {block_size_N}x{block_size_N}, Quality: {quality_level}")