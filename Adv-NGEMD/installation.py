!pip install torch torchvision numpy matplotlib pillow
!pip install torch torchvision Pillow numpy
#!pip install --upgrade Pillow
!pip install torch==1.11.0 torchvision==0.12.0
!pip install numpy pandas matplotlib
!pip install imageio
!apt-get install -y git
!git clone https://github.com/ha-rafiee/Steganalysis
%cd Deep-Steganalysis
!pip install -r requirements.txt
!pip install numpy pillow matplotlib torch torchvision scikit-image
from google.colab import drive
drive.mount('/content/drive')