**An Unsupervised Contrastive Transformer Network for Sparse Geochemical Anomaly Identification**

The proposed method combines contrastive learning with a Vision Transformer (ViT) architecture. Discriminative feature representations are learned from multi-element geochemical data through unsupervised representation learning. Geochemical anomalies are identified based on spatial feature representations.
The main training module (main_CTGA.py) is built on a MoCo-based ViT framework. Unsupervised contrastive learning is performed to extract latent feature representations from multi-element geochemical data. No anomaly labels are required during training. The embedding space structure is optimized through contrastive loss.
The detection module (detection.py) uses the pretrained encoder to extract feature representations of samples. Anomaly scores are generated through feature distance computation. Model performance is evaluated using metrics such as ROC-AUC.
The vit_pytorch module implements Patch Embedding, multi-head self-attention, and Transformer encoder blocks. Dependencies among geochemical elements are modeled through these components. The utils module provides data preprocessing and evaluation metric computation.
The LDPC module(LDPC.py) is used for positive sample construction in experiments. 

## 1.Structure
├─ vit_pytorch.py 
├─ main_CTGA.py 
├─ detection.py  
├─ utils.py 
├─ moco/
│ ├─ init.py 
│ ├─ builder.py 
│ └─ loader.py
└─ LDPC.py 

## 2. Environment
- Python 3.9
- PyTorch 2.7.1(CUDA 11.8 recommended for GPU acceleration)

## 3. Requirements
- NumPy
- SciPy
- scikit-learn
- Pillow
- timm 

