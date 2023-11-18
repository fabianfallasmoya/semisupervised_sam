# Run commands:
- python>=3.8
- pip3 install torch torchvision torchaudio 
- pip install timm
- pip install PyYAML
- pip install pycocotools
- pip install omegaconf
- pip install psutil
- pip install scikit-learn

## For FastSAM 
- pip install git+https://github.com/openai/CLIP.git
- pip install ultralytics == 8.0.120

# Uninstall previous version of Open CV and reinstall the proper one
- pip uninstall opencv-contrib-python opencv-python
- pip install opencv-contrib-python

# To install SAM
- pip install git+https://github.com/facebookresearch/segment-anything.git

# To install FastSAM
- pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git

# To install MobileSAM
- pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Other resources
- official Github repo: https://github.com/facebookresearch/segment-anything
- An example of SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
- An example of how to fine-tune SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
- An example of FastSAM: https://blog.roboflow.com/what-is-fastsam/