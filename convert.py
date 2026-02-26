import os
import subprocess
import sys

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# 1. Check and Install tensorflowjs
try:
    import tensorflowjs
    print("✅ tensorflowjs is already installed.")
except ImportError:
    print("⚠️ tensorflowjs not found. Attempting to auto-install...")
    try:
        install("tensorflowjs")
        install("h5py") 
    except Exception as e:
        print(f"❌ Failed to install packages: {e}")
        print("👉 Please try running: pip install tensorflowjs h5py")
        sys.exit(1)

# 2. Define Paths
model_path = "RespiScan_Final_Model.h5"
output_dir = "public/AI"

if not os.path.exists(model_path):
    print(f"❌ Error: Source file '{model_path}' not found in current directory!")
    sys.exit(1)

# 3. Execute Conversion
print(f"🚀 Converting '{model_path}' to '{output_dir}' (Format: tfjs_graph_model)...")

# Construct command using python module to avoid path issues
cmd = [
    sys.executable, "-m", "tensorflowjs.converters.converter",
    "--input_format", "keras",
    "--output_format", "tfjs_graph_model", # Using Graph Model for maximum compatibility
    "--weight_shard_size_bytes", "4194304",
    model_path,
    output_dir
]

try:
    subprocess.check_call(cmd)
    print("\n✅ Conversion SUCCESSFUL!")
    print(f"📂 Files saved to: {os.path.abspath(output_dir)}")
    print("👉 Now refresh your browser (Ctrl+F5) to test.")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Conversion FAILED with error code {e.returncode}")
    print("Try running this command manually:")
    print(" ".join(cmd))
