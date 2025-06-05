import os
import subprocess
import shutil
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import time

# --- Project Settings ---
BASE_PROJECT_DIR = r"C:\Users\Sazgar\Desktop\multuimedia\project\JPEG\colorised" # Your main project path
INPUT_IMAGES_DIR_NAME = "input_images"
RESULTS_BASE_DIR_NAME = "results"
PYTHON_EXECUTABLE = "python"

ALGORITMO_JPEG_SCRIPT = os.path.join(BASE_PROJECT_DIR, "Algoritmo_jpeg.py")
GENERADOR_HUFFMAN_SCRIPT = os.path.join(BASE_PROJECT_DIR, "generador_huffman_code.py")
DESCOMPRESOR_SCRIPT = os.path.join(BASE_PROJECT_DIR, "descompresor.py")
IDTC_SCRIPT = os.path.join(BASE_PROJECT_DIR, "idtc.py")

INPUT_IMAGES_FULL_PATH = os.path.join(BASE_PROJECT_DIR, INPUT_IMAGES_DIR_NAME)
RESULTS_BASE_FULL_PATH = os.path.join(BASE_PROJECT_DIR, RESULTS_BASE_DIR_NAME)

# --- Helper Functions ---
def get_image_size_kb(image_path):
    if not os.path.exists(image_path):
        return 0
    return os.path.getsize(image_path) / 1024.0

def calculate_metrics(original_image_path, compressed_image_path):
    try:
        img_orig_pil = Image.open(original_image_path)
        img_comp_pil = Image.open(compressed_image_path)
        img_orig_rgb = img_orig_pil.convert("RGB")
        img_comp_rgb = img_comp_pil.convert("RGB")

        if img_orig_rgb.size != img_comp_rgb.size:
            img_comp_rgb = img_comp_rgb.resize(img_orig_rgb.size, Image.Resampling.LANCZOS)

        img_orig_arr = np.array(img_orig_rgb)
        img_comp_arr = np.array(img_comp_rgb)

        psnr = peak_signal_noise_ratio(img_orig_arr, img_comp_arr, data_range=255)
        win_size = min(7, img_orig_arr.shape[0], img_orig_arr.shape[1])
        if win_size < 2 : ssim = np.nan
        else:
            if win_size % 2 == 0: win_size -=1
            ssim = structural_similarity(img_orig_arr, img_comp_arr, multichannel=True, channel_axis=-1, data_range=255, win_size=win_size)
        return psnr, ssim
    except FileNotFoundError:
        print(f"    [!] ERROR: Image file not found for metrics: {original_image_path} or {compressed_image_path}")
        return np.nan, np.nan
    except Exception as e:
        print(f"    [!] ERROR calculating metrics for {compressed_image_path}: {e}")
        return np.nan, np.nan

def run_compression_pipeline(input_image_path,
                             output_subdir,
                             quality, subsampling, block_size,
                             is_standard_mode=False,
                             run_identifier=""):
    scenario_name = run_identifier if run_identifier else f"Q{quality}_Sub{subsampling.replace(':', '')}_B{block_size}"
    if is_standard_mode:
        scenario_name = "Standard_Q50_Sub420_B8"
        q_matrix_type = "StandardJPEG"
    elif block_size == 8 and quality == 50: # Dynamic but happens to be standard Q params
        q_matrix_type = "StandardJPEG_via_Dynamic" # Or just "StandardJPEG" if Algoritmo_jpeg.py forces it
    else:
        q_matrix_type = f"DynamicHVS_B{block_size}"


    print(f"  [*] Running Scenario: {scenario_name} for {os.path.basename(input_image_path)} -> in {os.path.basename(output_subdir)}")

    cmd_algo = [
        PYTHON_EXECUTABLE, ALGORITMO_JPEG_SCRIPT,
        "--input_image", input_image_path,
        "--output_dir", output_subdir,
    ]
    if is_standard_mode: # This flag tells Algoritmo_jpeg.py to use its internal standard settings
        cmd_algo.append("--standard_mode")
    else:
        cmd_algo.extend([
            "--quality", str(quality),
            "--subsampling", subsampling,
            "--block_size", str(block_size)
        ])

    process_algo = subprocess.run(cmd_algo, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if process_algo.returncode != 0:
        print(f"    [!] ERROR running {ALGORITMO_JPEG_SCRIPT} for {scenario_name}:\nSTDOUT:\n{process_algo.stdout}\nSTDERR:\n{process_algo.stderr}")
        return None, None, q_matrix_type # Return q_matrix_type even on failure for record

    original_cwd = os.getcwd()
    os.chdir(output_subdir)

    expected_dct_file = "dct.txt"
    expected_result_file = "result.txt"
    expected_dims_file = "dims.txt"
    expected_q_y_file = "q_matrix_y_used.txt"
    expected_q_c_file = "q_matrix_c_used.txt"

    if not all(os.path.exists(f) for f in [expected_dct_file, expected_result_file, expected_dims_file, expected_q_y_file, expected_q_c_file]):
        print(f"    [!] ERROR: Required output files from Algoritmo_jpeg.py not found in {output_subdir}")
        os.chdir(original_cwd)
        return None, None, q_matrix_type

    cmd_huff = [PYTHON_EXECUTABLE, GENERADOR_HUFFMAN_SCRIPT, "-f", expected_dct_file,
                "--prob_file", expected_result_file,
                "--codes_out", "codigos.txt",
                "--compressed_out", "comprimido.dat"]
    process_huff = subprocess.run(cmd_huff, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if process_huff.returncode != 0:
        print(f"    [!] ERROR running {GENERADOR_HUFFMAN_SCRIPT} for {scenario_name}:\nSTDOUT:\n{process_huff.stdout}\nSTDERR:\n{process_huff.stderr}")
        os.chdir(original_cwd)
        return None, None, q_matrix_type

    cmd_desc = [PYTHON_EXECUTABLE, DESCOMPRESOR_SCRIPT, "-f", "comprimido.dat",
                "--codes_file", "codigos.txt",
                "--matrix_out", "matrix.txt"]
    process_desc = subprocess.run(cmd_desc, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if process_desc.returncode != 0:
        print(f"    [!] ERROR running {DESCOMPRESOR_SCRIPT} for {scenario_name}:\nSTDOUT:\n{process_desc.stdout}\nSTDERR:\n{process_desc.stderr}")
        os.chdir(original_cwd)
        return None, None, q_matrix_type

    temp_decompressed_name_in_idtc = "temp_decompressed_image.jpg"
    cmd_idtc = [PYTHON_EXECUTABLE, IDTC_SCRIPT,
                "--matrix_file", "matrix.txt",
                "--dims_file", expected_dims_file,
                "--q_y_file", expected_q_y_file,
                "--q_c_file", expected_q_c_file,
                "--output_image", temp_decompressed_name_in_idtc]
    process_idtc = subprocess.run(cmd_idtc, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if process_idtc.returncode != 0:
        print(f"    [!] ERROR running {IDTC_SCRIPT} for {scenario_name}:\nSTDOUT:\n{process_idtc.stdout}\nSTDERR:\n{process_idtc.stderr}")
        os.chdir(original_cwd)
        return None, None, q_matrix_type

    os.chdir(original_cwd)

    final_temp_compressed_path = os.path.join(output_subdir, temp_decompressed_name_in_idtc)
    if not os.path.exists(final_temp_compressed_path):
        print(f"    [!] ERROR: Final decompressed image by IDTC ({final_temp_compressed_path}) not found.")
        return None, None, q_matrix_type

    base_img_name, _ = os.path.splitext(os.path.basename(input_image_path))
    final_output_filename = f"{base_img_name}_{scenario_name}.jpg"
    final_output_path = os.path.join(output_subdir, final_output_filename)

    try:
        if os.path.exists(final_output_path): os.remove(final_output_path)
        os.rename(final_temp_compressed_path, final_output_path)
    except OSError as e:
        print(f"    [!] ERROR renaming output file {final_temp_compressed_path} to {final_output_path}: {e}")
        if os.path.exists(final_temp_compressed_path): final_output_path = final_temp_compressed_path
        else: return None, None, q_matrix_type

    compressed_size_kb = get_image_size_kb(final_output_path)
    return final_output_path, compressed_size_kb, q_matrix_type

# --- Test Scenarios (Full as requested) ---
qualities_to_test = [10, 25, 50, 75, 90]
subsampling_modes_to_test = ["4:4:4", "4:2:2", "4:2:0"]
block_sizes_to_test = [4, 8, 16]

BASE_Q_FOR_SUB_BLOCK = 75
BASE_SUB_FOR_Q_BLOCK = "4:2:0"
BASE_BLOCK_FOR_Q_SUB = 8

key_combo_scenarios = [
    {"quality": 95, "subsampling": "4:4:4", "block_size": 8, "id": "VeryHighQ_444_B8"},
    {"quality": 60, "subsampling": "4:2:2", "block_size": 16, "id": "MidHighQ_422_B16"},
    {"quality": 40, "subsampling": "4:2:0", "block_size": 4, "id": "MidLowQ_420_B4"},
    {"quality": 75, "subsampling": "4:4:4", "block_size": 16, "id": "GoodQ_444_B16"},
    {"quality": 20, "subsampling": "4:2:0", "block_size": 8, "id": "VeryLowQ_420_B8"},
]

def main_runner():
    if not os.path.exists(INPUT_IMAGES_FULL_PATH):
        print(f"❌ ERROR: Input images folder '{INPUT_IMAGES_FULL_PATH}' not found. Please create it and add images.")
        return

    os.makedirs(RESULTS_BASE_FULL_PATH, exist_ok=True)
    all_results_data = []
    image_files_to_process = [f for f in os.listdir(INPUT_IMAGES_FULL_PATH) if os.path.isfile(os.path.join(INPUT_IMAGES_FULL_PATH, f))]

    if not image_files_to_process:
        print(f"⚠️ WARNING: No image files found in '{INPUT_IMAGES_FULL_PATH}'.")
        return

    for image_file_name_with_ext in image_files_to_process:
        input_image_path = os.path.join(INPUT_IMAGES_FULL_PATH, image_file_name_with_ext)
        try: Image.open(input_image_path).verify()
        except Exception as img_err:
            print(f"⚠️ WARNING: Skipping invalid image '{image_file_name_with_ext}': {img_err}")
            continue
        
        print(f"\n==================================================================")
        print(f"🖼️  Processing Image: {image_file_name_with_ext}...")
        print(f"==================================================================")

        image_name_no_ext = os.path.splitext(image_file_name_with_ext)[0]
        image_overall_output_dir = os.path.join(RESULTS_BASE_FULL_PATH, image_name_no_ext)
        os.makedirs(image_overall_output_dir, exist_ok=True)

        original_size_kb = get_image_size_kb(input_image_path)
        if original_size_kb == 0:
            print(f"  [!] WARNING: Original image size is zero for '{image_file_name_with_ext}'. Skipping.")
            continue

        print(f"\n  --- 🏁 Running Standard Mode for {image_file_name_with_ext} ---")
        standard_scenario_output_dir = os.path.join(image_overall_output_dir, "Standard_Q50_Sub420_B8_RUN")
        os.makedirs(standard_scenario_output_dir, exist_ok=True)

        start_time = time.time()
        standard_output_path, standard_compressed_size, standard_q_type = run_compression_pipeline(
            input_image_path, standard_scenario_output_dir,
            quality=50, subsampling="4:2:0", block_size=8, is_standard_mode=True
        )
        elapsed_time = time.time() - start_time

        standard_psnr, standard_ssim, standard_comp_ratio = np.nan, np.nan, np.nan
        if standard_output_path and standard_compressed_size is not None and standard_compressed_size > 0:
            standard_psnr, standard_ssim = calculate_metrics(input_image_path, standard_output_path)
            standard_comp_ratio = original_size_kb / standard_compressed_size if standard_compressed_size > 0 else np.inf
            all_results_data.append({
                "Image": image_file_name_with_ext, "ScenarioID": "Standard_Q50_Sub420_B8",
                "Quality": 50, "Subsampling": "4:2:0", "BlockSize": 8, "QMatrixType": standard_q_type,
                "OriginalSizeKB": original_size_kb, "CompressedSizeKB": standard_compressed_size,
                "CompressionRatio": standard_comp_ratio, "TimeSec": elapsed_time,
                "PSNR_vs_Original": standard_psnr, "SSIM_vs_Original": standard_ssim,
                "PSNR_vs_Standard": np.nan, "SSIM_vs_Standard": np.nan
            })
            print(f"    📊 Standard: PSNR={standard_psnr:.2f}, SSIM={standard_ssim:.4f}, CR={standard_comp_ratio:.2f}x, Size={standard_compressed_size:.2f}KB, Time={elapsed_time:.2f}s")
        else:
            print(f"    [!] ERROR running standard mode for {image_file_name_with_ext}.")

        scenarios_to_run_for_this_image = []
        # Effect of Quality
        for q_val in qualities_to_test:
            scenarios_to_run_for_this_image.append({"quality": q_val, "subsampling": BASE_SUB_FOR_Q_BLOCK, "block_size": BASE_BLOCK_FOR_Q_SUB, "id_prefix": "Effect_Q"})
        # Effect of Subsampling
        for sub_val in subsampling_modes_to_test:
            scenarios_to_run_for_this_image.append({"quality": BASE_Q_FOR_SUB_BLOCK, "subsampling": sub_val, "block_size": BASE_BLOCK_FOR_Q_SUB, "id_prefix": "Effect_Sub"})
        # Effect of Block Size
        for bs_val in block_sizes_to_test:
            scenarios_to_run_for_this_image.append({"quality": BASE_Q_FOR_SUB_BLOCK, "subsampling": BASE_SUB_FOR_Q_BLOCK, "block_size": bs_val, "id_prefix": "Effect_Block"})
        # Key combinations
        for combo in key_combo_scenarios:
            scenarios_to_run_for_this_image.append(combo)

        final_dynamic_scenarios = []
        seen_params_for_dynamic = set()
        # Add standard params to avoid re-running it if it appears in dynamic scenarios
        if standard_output_path: # only if standard run was successful
            seen_params_for_dynamic.add((50, "4:2:0", 8)) 
            
        for scn_params in scenarios_to_run_for_this_image:
            params_tuple = (scn_params['quality'], scn_params['subsampling'], scn_params['block_size'])
            if params_tuple not in seen_params_for_dynamic:
                seen_params_for_dynamic.add(params_tuple)
                final_dynamic_scenarios.append(scn_params)
        
        print(f"\n  --- ⚙️  Running {len(final_dynamic_scenarios)} Dynamic Scenarios for {image_file_name_with_ext} ---")
        for scenario_params in final_dynamic_scenarios:
            q = scenario_params["quality"]
            sub = scenario_params["subsampling"]
            bs = scenario_params["block_size"]
            scenario_id_str = scenario_params.get("id", f"Q{q}_Sub{sub.replace(':', '')}_B{bs}")
            
            scenario_output_dir = os.path.join(image_overall_output_dir, scenario_id_str)
            os.makedirs(scenario_output_dir, exist_ok=True)

            start_time = time.time()
            output_path, compressed_size, q_type_run = run_compression_pipeline(
                input_image_path, scenario_output_dir, q, sub, bs, run_identifier=scenario_id_str
            )
            elapsed_time = time.time() - start_time
            if output_path and compressed_size is not None and compressed_size > 0:
                psnr_o, ssim_o = calculate_metrics(input_image_path, output_path)
                psnr_s, ssim_s = (calculate_metrics(standard_output_path, output_path) if standard_output_path else (np.nan, np.nan))
                cr = original_size_kb / compressed_size if compressed_size > 0 else np.inf
                all_results_data.append({"Image": image_file_name_with_ext, "ScenarioID": scenario_id_str,
                                         "Quality": q, "Subsampling": sub, "BlockSize": bs, "QMatrixType": q_type_run,
                                         "OriginalSizeKB": original_size_kb, "CompressedSizeKB": compressed_size,
                                         "CompressionRatio": cr, "TimeSec": elapsed_time,
                                         "PSNR_vs_Original": psnr_o, "SSIM_vs_Original": ssim_o,
                                         "PSNR_vs_Standard": psnr_s, "SSIM_vs_Standard": ssim_s})
                print(f"    📊 {scenario_id_str}: PSNR_O={psnr_o:.2f}, SSIM_O={ssim_o:.4f}, CR={cr:.2f}x, Size={compressed_size:.2f}KB, Time={elapsed_time:.2f}s")
            else:
                print(f"    [!] FAILED SCENARIO: {scenario_id_str}")

    if all_results_data:
        df_results = pd.DataFrame(all_results_data)
        results_csv_path = os.path.join(RESULTS_BASE_FULL_PATH, "full_project_compression_results.csv")
        try:
            df_results.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n✅ Results saved to '{results_csv_path}'")
        except Exception as e:
            print(f"❌ ERROR saving CSV results: {e}")
        
        plot_results(df_results, RESULTS_BASE_FULL_PATH)
    else:
        print("\n⚠️ No results were generated to save or plot.")

def plot_results(df, base_results_dir_for_plots):
    if df.empty: print("No data to plot."); return
    print("\n📊 Generating plots...")
    overall_plots_dir = os.path.join(base_results_dir_for_plots, "overall_project_plots")
    os.makedirs(overall_plots_dir, exist_ok=True)

    for image_name, group_df_img in df.groupby("Image"):
        image_name_no_ext = os.path.splitext(image_name)[0]
        image_plot_dir = os.path.join(overall_plots_dir, image_name_no_ext)
        os.makedirs(image_plot_dir, exist_ok=True)

        # --- PSNR vs Quality ---
        df_q_effect = group_df_img[
            (group_df_img["BlockSize"] == BASE_BLOCK_FOR_Q_SUB) &
            (group_df_img["Subsampling"] == BASE_SUB_FOR_Q_BLOCK)
        ].copy()
        if not df_q_effect.empty and "Quality" in df_q_effect.columns:
            df_q_effect.loc[:, "Quality"] = pd.to_numeric(df_q_effect["Quality"])
            df_q_effect = df_q_effect.sort_values(by="Quality")
            plt.figure(figsize=(10,6)); plt.plot(df_q_effect["Quality"], df_q_effect["PSNR_vs_Original"], marker='o', label="PSNR vs Original")
            if "PSNR_vs_Standard" in df_q_effect.columns and df_q_effect["PSNR_vs_Standard"].notna().any():
                 plt.plot(df_q_effect["Quality"], df_q_effect["PSNR_vs_Standard"], marker='x', linestyle='--', label="PSNR vs Standard Output")
            plt.xlabel("Quality Level"); plt.ylabel("PSNR (dB)")
            plt.title(f"PSNR vs Quality for {image_name}\n(Block={BASE_BLOCK_FOR_Q_SUB}, Subsampling={BASE_SUB_FOR_Q_BLOCK})")
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(image_plot_dir, f"{image_name_no_ext}_psnr_vs_quality.png")); plt.close()

        # --- Compression Ratio vs Block Size ---
        df_b_effect = group_df_img[
            (group_df_img["Quality"] == BASE_Q_FOR_SUB_BLOCK) &
            (group_df_img["Subsampling"] == BASE_SUB_FOR_Q_BLOCK)
        ].copy()
        if not df_b_effect.empty and "BlockSize" in df_b_effect.columns:
            df_b_effect.loc[:, "BlockSize"] = pd.to_numeric(df_b_effect["BlockSize"])
            df_b_effect = df_b_effect.sort_values(by="BlockSize")
            plt.figure(figsize=(10,6)); block_str = df_b_effect["BlockSize"].astype(str) + "x" + df_b_effect["BlockSize"].astype(str)
            plt.bar(block_str, df_b_effect["CompressionRatio"])
            plt.xlabel("DCT Block Size"); plt.ylabel("Compression Ratio (Original/Compressed)")
            plt.title(f"Compression Ratio vs Block Size for {image_name}\n(Quality={BASE_Q_FOR_SUB_BLOCK}, Subsampling={BASE_SUB_FOR_Q_BLOCK})")
            plt.grid(axis='y'); plt.savefig(os.path.join(image_plot_dir, f"{image_name_no_ext}_cr_vs_blocksize.png")); plt.close()

        # --- SSIM vs Subsampling ---
        df_s_effect = group_df_img[
            (group_df_img["Quality"] == BASE_Q_FOR_SUB_BLOCK) &
            (group_df_img["BlockSize"] == BASE_BLOCK_FOR_Q_SUB)
        ].copy()
        if not df_s_effect.empty and "Subsampling" in df_s_effect.columns:
            order = ["4:4:4", "4:2:2", "4:2:0"]
            df_s_effect = df_s_effect[df_s_effect["Subsampling"].isin(order)]
            if not df_s_effect.empty:
                df_s_effect.loc[:, "Subsampling"] = pd.Categorical(df_s_effect["Subsampling"], categories=order, ordered=True)
                df_s_effect = df_s_effect.sort_values(by="Subsampling")
                if not df_s_effect.empty:
                    plt.figure(figsize=(10,6)); plt.bar(df_s_effect["Subsampling"].astype(str), df_s_effect["SSIM_vs_Original"])
                    plt.xlabel("Chroma Subsampling Type"); plt.ylabel("SSIM vs Original")
                    plt.title(f"SSIM vs Subsampling for {image_name}\n(Quality={BASE_Q_FOR_SUB_BLOCK}, BlockSize={BASE_BLOCK_FOR_Q_SUB})")
                    plt.grid(axis='y'); plt.savefig(os.path.join(image_plot_dir, f"{image_name_no_ext}_ssim_vs_subsampling.png")); plt.close()
    print("✅ Plot generation completed.")

if __name__ == "__main__":
    if not os.path.exists(INPUT_IMAGES_FULL_PATH):
        os.makedirs(INPUT_IMAGES_FULL_PATH)
        print(f"Directory '{INPUT_IMAGES_FULL_PATH}' created. Please place your input images there.")
    main_runner()