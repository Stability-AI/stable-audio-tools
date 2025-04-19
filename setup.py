from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='stable-audio-tools',
    version='0.0.18',  # Keep the version from the main branch
    url='https://github.com/Stability-AI/stable-audio-tools.git',
    author='Stability AI',
    author_email='info@stability.ai',
    description='Training and inference tools for generative audio models from Stability AI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
    install_requires=[
        'alias-free-torch==0.0.6',
        'auraloss==0.4.0',
        'descript-audio-codec==1.0.0',
        'einops',
        'einops-exts',
        'ema-pytorch==0.2.3',
        'encodec==0.1.1',
        'gradio>=3.42.0',
        'huggingface_hub',
        'importlib-resources==5.12.0',
        'k-diffusion==0.1.1',
        'laion-clap==1.1.4',
        'local-attention==1.8.6',
        'pandas==2.0.2',
        'prefigure==0.0.9',
        'pytorch_lightning==2.1.0',
        'PyWavelets==1.4.1',
        'safetensors',
        'sentencepiece==0.1.99',
        'torch>=2.0.1',
        'torchaudio>=2.0.2',
        'torchmetrics==0.11.4',
        'tqdm',
        'transformers',
        'v-diffusion-pytorch==0.0.2',
        'vector-quantize-pytorch==1.14.41',
        'wandb==0.15.4',
        'webdataset==0.2.100',
        'x-transformers<1.27.0'
    ],
    entry_points={
        'console_scripts': [
            'stable-audio-tools=stable_audio_tools.cli:main',
        ],
    },
)
