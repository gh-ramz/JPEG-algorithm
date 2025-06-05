import sys
import os
import operator
import argparse

def huffman_compress(input_dct_file, input_prob_file, output_huff_codes_file, output_compressed_data_file):
    probabilities = {}
    try:
        with open(input_prob_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    symbol = parts[0]
                    try: probability = float(parts[1]); probabilities[symbol] = probability
                    except ValueError: print(f"    [!] WARNING (huff_compress): Could not parse prob for '{symbol}' in {input_prob_file}.")
    except FileNotFoundError:
        print(f"    [!] ERROR (huff_compress): Probability file '{input_prob_file}' not found."); return False
    except Exception as e:
        print(f"    [!] ERROR (huff_compress) reading prob file '{input_prob_file}': {e}"); return False

    if not probabilities:
        print(f"    [!] ERROR (huff_compress): No probabilities loaded from '{input_prob_file}'."); return False

    code_table = generate_huffman_codes(probabilities)
    if not code_table:
        if len(probabilities) == 1:
            single_symbol = list(probabilities.keys())[0]
            code_table = {single_symbol: "0"}
        else:
            print("    [!] ERROR (huff_compress): Huffman code table generation failed."); return False
    
    try:
        save_codebook(code_table, output_huff_codes_file)
    except Exception as e:
        print(f"    [!] ERROR (huff_compress) saving codebook to '{output_huff_codes_file}': {e}"); return False

    try:
        with open(input_dct_file, 'r', encoding='utf-8') as infile, open(output_compressed_data_file, 'wb') as outfile:
            text_data = infile.read().strip()
            if not text_data:
                # print(f"    [!] WARNING (huff_compress): Input DCT file '{input_dct_file}' is empty.")
                padded_text = pad_encoded_text("")
            else:
                encoded_text = encode_text(code_table, text_data)
                if not encoded_text and text_data : print("    [!] WARNING (huff_compress): Encoded text is empty.")
                padded_text = pad_encoded_text(encoded_text)
            
            if padded_text is None: print("    [!] ERROR (huff_compress): Padding failed."); return False
            binary_array = generate_bit_array(padded_text)
            if binary_array is None: print("    [!] ERROR (huff_compress): Bit array generation failed."); return False
            outfile.write(bytes(binary_array))
    except FileNotFoundError:
        print(f"    [!] ERROR (huff_compress): Input DCT file '{input_dct_file}' not found."); return False
    except Exception as e:
        print(f"    [!] ERROR (huff_compress) during file operations or encoding: {e}"); return False
    
    print(f"    [+] INFO (huff_compress): Huffman compression successful. Output: '{os.path.basename(output_compressed_data_file)}'")
    return True

def generate_huffman_codes(prob_dict):
    if not prob_dict: return {}
    valid_probs = {k: v for k, v in prob_dict.items() if v > 0}
    if not valid_probs: return {}
    if len(valid_probs) == 1:
        key = list(valid_probs.keys())[0]
        return {key: '0'}

    nodes = [{'symbol': symbol, 'prob': prob, 'is_leaf': True} for symbol, prob in valid_probs.items()]
    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x['prob'])
        left = nodes.pop(0); right = nodes.pop(0)
        new_node_prob = left['prob'] + right['prob']
        new_node = {'symbol': None, 'prob': new_node_prob, 'left': left, 'right': right, 'is_leaf': False}
        nodes.append(new_node)
    
    codes = {}
    def build_codes_recursive(node, current_code=""):
        if node is None: return
        if node.get('is_leaf', False):
            codes[node['symbol']] = current_code if current_code else "0"
            return
        if node.get('left'): build_codes_recursive(node['left'], current_code + "0")
        if node.get('right'): build_codes_recursive(node['right'], current_code + "1")
    
    if nodes: build_codes_recursive(nodes[0])
    return codes

def save_codebook(codebook, filepath):
    with open(filepath, "w", encoding='utf-8') as file:
        for symbol, code in codebook.items():
            file.write(str(symbol) + "\t" + code + "\n")

def encode_text(codebook, text_data_string):
    encoded_output = ""; missing_symbols_count = 0
    symbols = text_data_string.split()
    if not symbols and text_data_string:
        # print("    [!] WARNING (encode_text): Input text data resulted in no symbols after split.")
        return ""

    for symbol_str in symbols:
        if symbol_str in codebook: encoded_output += codebook[symbol_str]
        else:
            # if missing_symbols_count < 5: print(f"    [!] WARNING (encode_text): Symbol '{symbol_str}' not found in Huffman codebook. Skipping.")
            missing_symbols_count +=1
    if missing_symbols_count > 0: print(f"    [i] INFO (encode_text): Total missing symbols during encoding: {missing_symbols_count}.")
    return encoded_output

def pad_encoded_text(encoded_bit_string):
    if encoded_bit_string is None: return None
    padding_length = (8 - len(encoded_bit_string) % 8) % 8
    padded_string = encoded_bit_string + "0" * padding_length
    padding_info_str = f"{padding_length:08b}"
    return padding_info_str + padded_string

def generate_bit_array(padded_binary_string):
    if padded_binary_string is None: return None
    if len(padded_binary_string) % 8 != 0:
        print("    [!] ERROR (generate_bit_array): Binary string length not multiple of 8 for byte array conversion.")
        return None
    bit_array = bytearray()
    for i in range(0, len(padded_binary_string), 8):
        byte_segment = padded_binary_string[i:i+8]
        try: bit_array.append(int(byte_segment, 2))
        except ValueError: print(f"    [!] ERROR (generate_bit_array): Invalid byte segment '{byte_segment}'."); return None
    return bit_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huffman compresses a DCT coefficient file.")
    parser.add_argument("-f", "--dct_file", required=True, help="Path to the input DCT coefficients file.")
    parser.add_argument("--prob_file", default="result.txt", help="Path to the probabilities file.")
    parser.add_argument("--codes_out", default="codigos.txt", help="Path for the output Huffman codes file.")
    parser.add_argument("--compressed_out", default="comprimido.dat", help="Path for the output compressed data file.")
    args = parser.parse_args()
    
    if not huffman_compress(args.dct_file, args.prob_file, args.codes_out, args.compressed_out):
        print("❌ Huffman compression failed when run independently.")
        sys.exit(1)
    # else:
        # print("✅ Huffman compression successful (when run independently).")