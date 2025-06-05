import sys
import argparse
import os

def decompress(input_compressed_data_file, input_huff_codes_file, output_matrix_file):
    try:
        with open(input_compressed_data_file, 'rb') as file:
            bit_string = ""
            byte = file.read(1)
            while byte != b'':
                byte_value = ord(byte)
                bits = bin(byte_value)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)
    except FileNotFoundError:
        print(f"    [!] ERROR (decompress): Compressed data file '{input_compressed_data_file}' not found."); return False
    except Exception as e:
        print(f"    [!] ERROR (decompress) reading compressed file: {e}"); return False

    if not bit_string:
        # print(f"    [!] WARNING (decompress): Compressed file '{input_compressed_data_file}' is empty or unreadable.")
        try:
            with open(output_matrix_file, "w", encoding='utf-8') as file: file.write("")
            return True
        except IOError: return False

    encoded_text_padded = bit_string
    encoded_text = remove_bit_padding(encoded_text_padded)
    if encoded_text is None: print("    [!] ERROR (decompress): Removing padding failed."); return False

    decompressed_text_data = decode_text(encoded_text, input_huff_codes_file)
    if decompressed_text_data is None : print("    [!] ERROR (decompress): Decoding text failed."); return False

    try:
        with open(output_matrix_file, "w", encoding='utf-8') as file:
            file.write(decompressed_text_data)
    except IOError as e:
        print(f"    [!] ERROR (decompress) writing to output matrix file '{output_matrix_file}': {e}"); return False
    
    print(f"    [+] INFO (decompress): Decompression successful. Output: '{os.path.basename(output_matrix_file)}'")
    return True

def remove_bit_padding(encoded_text_with_padding):
    if len(encoded_text_with_padding) < 8:
        print("    [!] ERROR (remove_padding): Encoded text too short for padding info.")
        return None
    padding_info = encoded_text_with_padding[:8]
    try: extra_padding = int(padding_info, 2)
    except ValueError: print(f"    [!] ERROR (remove_padding): Invalid padding info bits '{padding_info}'."); return None

    encoded_text_main_data = encoded_text_with_padding[8:]
    if extra_padding > 0:
        if len(encoded_text_main_data) < extra_padding:
            print(f"    [!] ERROR (remove_padding): Encoded text (len {len(encoded_text_main_data)}) "
                  f"shorter than declared padding ({extra_padding}).")
            return None
        return encoded_text_main_data[:-extra_padding]
    return encoded_text_main_data

def decode_text(encoded_bit_stream, huff_codes_filepath):
    code_to_symbol_map = {}
    try:
        with open(huff_codes_filepath, "r", encoding='utf-8') as code_file:
            for line in code_file:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    symbol = parts[0]; code = parts[1]
                    code_to_symbol_map[code] = symbol
    except FileNotFoundError:
        print(f"    [!] ERROR (decode_text): Huffman codes file '{huff_codes_filepath}' not found."); return None
    except Exception as e:
        print(f"    [!] ERROR (decode_text) reading codes file '{huff_codes_filepath}': {e}"); return None

    if not code_to_symbol_map:
        print(f"    [!] ERROR (decode_text): Codebook from '{huff_codes_filepath}' is empty."); return None

    current_code = ""; decoded_numbers = []
    for bit in encoded_bit_stream:
        current_code += bit
        if current_code in code_to_symbol_map:
            decoded_numbers.append(code_to_symbol_map[current_code])
            current_code = ""
    if current_code: print(f"    [!] WARNING (decode_text): Trailing bits '{current_code}' could not be decoded.")
    return " ".join(decoded_numbers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompresses a Huffman encoded file.")
    parser.add_argument("-f", "--compressed_file", required=True, help="Path to the input compressed data file.")
    parser.add_argument("--codes_file", default="codigos.txt", help="Path to the Huffman codes file.")
    parser.add_argument("--matrix_out", default="matrix.txt", help="Path for the output decompressed matrix file.")
    args = parser.parse_args()

    if not decompress(args.compressed_file, args.codes_file, args.matrix_out):
        print("❌ Decompression failed when run independently.")
        sys.exit(1)
    # else:
        # print(f"✅ File '{args.compressed_file}' successfully decompressed to '{args.matrix_out}' (when run independently).")