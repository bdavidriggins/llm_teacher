# Setup
sudo apt install -y python3 python3-pip python3-venv
python3 -m venv rag_env

# Activate Environment
source rag_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


