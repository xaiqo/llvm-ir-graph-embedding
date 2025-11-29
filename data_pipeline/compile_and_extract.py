import os
import subprocess
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

LLVM_BIN_DIR = "/usr/bin"
CLANG = "clang++-17"
OPT = "opt-17"
PASS_LIB = "llvm_pass/build/lib/GraphExtractor.so"

def compile_to_ir(src_file, output_dir, optimize=False):
    """Compiles C++ source to LLVM IR (.ll) with header injection"""
    filename = os.path.basename(src_file)
    name_no_ext = os.path.splitext(filename)[0]
    ir_file = os.path.join(output_dir, name_no_ext + ".ll")
    
    temp_src = os.path.join(output_dir, name_no_ext + "_temp.cpp")
    
    COMMON_HEADERS = """
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <iostream>
    #include <vector>
    #include <algorithm>
    #include <cmath>
    #include <map>
    #include <set>
    #include <queue>
    #include <stack>
    using namespace std;
    """
    
    try:
        with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
            original_code = f.read()
            
        with open(temp_src, 'w', encoding='utf-8') as f:
            f.write(COMMON_HEADERS + "\n" + original_code)
            
    except Exception as e:
        return None
    
    flags = [
        "-S", "-emit-llvm",                 
        "-fno-discard-value-names",
        "-Wno-everything", 
        "-c", temp_src,
        "-o", ir_file
    ]
    
    if optimize:
        flags.insert(0, "-O3")
    else:
        flags.insert(0, "-O0")
        
    cmd = [CLANG] + flags
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_src):
            os.remove(temp_src)
        return ir_file
    except subprocess.CalledProcessError:
        if os.path.exists(temp_src):
            os.remove(temp_src)
        return None

def extract_graph(ir_file, output_dir):
    """Runs the GraphExtractor pass on the IR file"""
    if not ir_file: return None
    
    filename = os.path.basename(ir_file)
    toon_file = os.path.join(output_dir, filename.replace(".ll", ".toon"))
    
    cmd = [
        OPT,
        "-load-pass-plugin", PASS_LIB,
        "-passes=extract-graph", 
        ir_file,
        "-disable-output" 
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        graph_data = result.stderr
        
        if "nodes[" in graph_data:
            with open(toon_file, "w") as f:
                f.write(graph_data)
            return toon_file
        else:
            return None
            
    except subprocess.CalledProcessError as e:
        return None

def process_file(args):
    src_file, ir_dir, graph_dir, optimize = args
    ir_file = compile_to_ir(src_file, ir_dir, optimize)
    if ir_file:
        return extract_graph(ir_file, graph_dir)
    return None

def main():
    parser = argparse.ArgumentParser(description="LLVM Graph Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Input directory containing .cpp files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument("--optimize", action="store_true", help="Use O3 optimization")
    args = parser.parse_args()

    ir_dir = os.path.join(args.output, "ir")
    graph_dir = os.path.join(args.output, "graphs")
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.input, "**/*.cpp"), recursive=True)
    print(f"Found {len(files)} source files.")

    tasks = [(f, ir_dir, graph_dir, args.optimize) for f in files]

    success_count = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks)))
        success_count = sum(1 for r in results if r is not None)

    print(f"Processing complete. Successfully extracted {success_count}/{len(files)} graphs.")

if __name__ == "__main__":
    main()
