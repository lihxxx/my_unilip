source /mnt/tidal-alsh01/dataset/zeus/lihongxiang/network.sh


echo "卸载原有包库"
pip freeze | xargs pip uninstall -y
echo "卸载完毕"
echo "安装环境"
apt install libgl1-mesa-glx -y

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
