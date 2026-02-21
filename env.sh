# Clone the repository
source /mnt/tidal-alsh01/dataset/zeus/lihongxiang/network.sh

# Install dependencies
# conda create -n UniLIP python=3.10
# conda activate UniLIP
echo "卸载原有包库"
pip freeze | xargs pip uninstall -y
echo "卸载完毕"
echo "安装环境"
apt install libgl1-mesa-glx -y
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install mmengine xtuner tqdm timm
# pip install diffusers==0.36.0 transformers==4.57.1
# pip install flash-attn --no-build-isolation
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
pip install -e .