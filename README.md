# 🖼️ Image Super-Resolution using Deep Learning

This project implements and compares deep learning models for enhancing low-resolution images. Using architectures such as **SRCNN**, **DSRCNN**, and **Real-ESRGAN**, we reconstruct high-resolution images with sharper details and reduced noise. Applications include medical imaging, surveillance, satellite imaging, and image restoration.

---

## 📁 Repository Structure

```
.
├── Model Weights/          # Trained model weights
├── Papers/                 # Research papers referenced           
│── srcnn.py                # Python scripts for models (note: converted from ipynb, structure is mixed)
│── dsr_cnn.py
│── real_esrgan.ipynb
├── Dockerfile              # Docker setup for local training
├── README.md               
```

> ⚠️ The `.py` scripts were converted from Jupyter notebooks, so they may contain interleaved imports, definitions, and function calls.

---

## 🧠 Models Implemented

### 1. **SRCNN (Super-Resolution CNN)**

* Based on [Dong et al., arXiv:1501.00092](https://arxiv.org/pdf/1501.00092v3)
* Architecture: 9×9 → 5×5 → 5×5 convolutions
* Training: 20 epochs, batch size 32, learning rate `1e-4`
* Achieved PSNR: **31.42 dB**

### 2. **DSRCNN (Denoising SRCNN)**

* Extension of SRCNN with added denoising functionality
* Trained under identical settings
* Achieved PSNR: **31.58 dB**

### 3. **Real-ESRGAN**

* Based on [Wang et al., arXiv:2107.10833](https://arxiv.org/pdf/2107.10833v2)
* Used a pre-trained model for qualitative comparison
* PSNR not applicable, but visual results show enhanced perceptual quality

---

## 🗃️ Dataset

* **Source**: [ILSVRC2013 Validation Set](https://image-net.org/challenges/LSVRC/2013/)
* **Preprocessing**: Downsampled by 3× using bicubic interpolation to create LR-HR image pairs.
* 📆 **Dataset Access**: [Google Drive Link](https://drive.google.com/file/d/1D8LZ7HRsW17D6lYaFFHAhCuST-wPfo37/view?usp=sharing) for faster downlaods

---

## 🐳 Docker Setup

Training was performed locally using Docker to ensure consistent environments and manage dependencies.

**Dockerfile:**

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0


RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

RUN pip install --upgrade pip setuptools setuptools-scm

# Install PyTorch, xFormers, and dependencies
RUN pip install torch \
    torchvision \ 
    torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    xformers 
RUN pip install transformers
RUN pip install fastapi uvicorn
RUN pip install trl 
RUN pip install peft 
RUN pip install accelerate 
RUN pip install bitsandbytes
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"


WORKDIR /app
VOLUME ["/app"]

# Run the model server
CMD ["bash"]
```

> 💡 This Dockerfile contains some extra installations and may be missing a few others. This is because the same image was reused across multiple model trainings. It was chosen for convenience since most major dependencies (such as CUDA, PyTorch, torchvision, etc.) were already included.

---

## 🧪 Results

* **Test Size**: 10% of dataset (\~2000 images)
* **Metric**: Peak Signal-to-Noise Ratio (PSNR)

| Model       | PSNR (dB) |
| ----------- | --------- |
| SRCNN       | 31.42     |
| DSRCNN      | 31.58     |
| Real-ESRGAN | N/A       |

<!-- ### 🖼️ Sample Outputs:

(Add training/epoch visualizations here) -->

---

## 🔧 Requirements

You can install dependencies using the following command:

```bash
pip install <package>
```

Minimal required packages:

* numpy
* matplotlib
* Pillow

---

## 📊 Future Work

* Implement perceptual loss for improved visual quality
* Fine-tune Real-ESRGAN on the same dataset for consistent benchmarking
* Evaluate additional super-resolution architectures
* Fine-tune models on larger and more diverse datasets

---

## 📝 Notes

* Initial development was done in **Google Colab**
* Local training was done via Docker to overcome Colab resource limits
* The code in `.py` files is partially unstructured due to conversion from notebooks for local execution

---

## ✨ Acknowledgements

* [Dong et al.](https://arxiv.org/abs/1501.00092) for SRCNN
* [Wang et al.](https://arxiv.org/abs/2107.10833) for Real-ESRGAN
* [CNN Github](https://github.com/titu1994/Image-Super-Resolution) for reference
* [ESR-GAN](https://github.com/xinntao/Real-ESRGAN) for reference and model weights
