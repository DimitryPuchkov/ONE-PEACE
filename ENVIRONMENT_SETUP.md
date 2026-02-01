#环境设置说明 (Environment Setup Instructions)

## Conda Environment Setup for ONE-PEACE-VIDEO-MODALITY

### Created Environment
- **Environment Name:** `one_peace_env`
- **Python Version:** 3.10
- **PyTorch Version:** 2.6.0 (CPU)

### To Activate the Environment

```bash
conda activate one_peace_env
```

### To Deactivate the Environment

```bash
conda deactivate
```

### Installation Summary

The conda environment has been created with Python 3.10 and essential packages installed successfully, including:

- **Core ML/DL:** torch, torchvision, torchaudio, numpy, scipy, pandas, scikit-learn
- **Transformers:** transformers, omegaconf, einops, bert_score
- **Vision:** opencv-python, Pillow, timm, imageio, av
- **Audio:** librosa, soundfile
- **Utilities:** wandb, tensorboardX, matplotlib, tqdm, etc.

### Modified Requirements

The following changes were made to resolve version conflicts and ensure compatibility:

1. **Removed/Replaced packages:**
   - ❌ `faiss==1.5.3` → ✅ `faiss-cpu==1.8.0` (old version unavailable)
   - ❌ `torch_xla==2.7.0` (removed - limited support and build issues)
   - ❌ `apex` (removed - requires CUDA compilation)
   - ❌ `tensorrt==10.11.0.33` (removed - NVIDIA-specific)
   - ❌ Removed yanked versions of omegaconf 2.0.x

2. **Updated versions:**
   - `numpy`: 2.2.6 → **1.26.4** (for compatibility with older packages)
   - `omegaconf`: 2.0.6 → **>=2.1.0** (yanked versions unavailable)

3. **Removed duplicate entries:**
   - Duplicate PyYAML, setuptools, tensorboardX entries

4. **Optional packages** (commented out due to build requirements):
   - `flash_attn==2.7.4.post1` - requires CUDA
   - `xformers==0.0.29.post3` - performance optimization
   - `deepspeed==0.16.7` - distributed training
   - `mmcv`, `mmdet`, `mmseg`, `mmaction` - complex dependencies

### Testing the Installation

```bash
# Activate environment
conda activate one_peace_env

# Test basic imports
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import cv2; print('OpenCV available')"
```

### Additional Setup

If you need to install optional packages:

```bash
# For GPU support (if available), modify PyTorch installation
conda activate one_peace_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For individual optional packages
pip install flash-attn==2.7.4.post1  # requires CUDA
```

### Python Version Compatibility

- **Tested:** Python 3.10
- **Why 3.10?** It provides the best balance between:
  - Compatibility with modern PyTorch (requires 3.9+)
  - Support for older packages that need compatibility
  - Wide package availability on PyPI

### Environment List

To see all conda environments:
```bash
conda env list
```

To remove environment if needed:
```bash
conda env remove -n one_peace_env
```

---

**Note:** Some packages from the original requirements.txt were excluded or modified due to:
- Version availability on PyPI
- Python 3.10 compatibility
- Build requirements (CUDA, compilation tools)
- Deprecated/yanked versions

The current setup provides all core functionality while ensuring stability and compatibility.
