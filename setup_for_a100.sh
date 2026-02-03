conda init && conda create -n multiplex-thinking python=3.10 -y
conda init && eval "$(conda shell.bash hook)" && conda activate multiplex-thinking
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install accelerate==1.10.1
pip install flash-attn==2.8.2 --no-build-isolation
pip install triton==3.3.1
pip install transformers==4.54.0
pip install sglang==0.4.9.post6
pip install sgl-kernel==0.2.8
pip install uv
pip install debugpy

bash install_more_pip.sh
bash setup.sh