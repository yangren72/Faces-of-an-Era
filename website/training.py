# codes modified from Jeff Heaton(https://github.com/jeffheaton) and NVIDIA github page(https://github.com/NVlabs/stylegan2-ada-pytorch)

# link to google drive and colab

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    COLAB = True
    print("Note: using Google CoLab")
    %tensorflow_version 2.x
except:
    print("Note: not using Google CoLab")
    COLAB = False


# install essential stuffs
!pip install torch==1.8.1 torchvision==0.9.1
!git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
!pip install ninja

# convert images
CMD = "python /content/stylegan2-ada-pytorch/dataset_tool.py "\
  "--source /content/drive/MyDrive/data/gan1/images/archiveW "\
  "--dest /content/drive/MyDrive/data/gan1/dataset/archive"

!{CMD}




# for initial training
import os


EXPERIMENTS = "/content/drive/MyDrive/data/gan1/experiments"
DATA = "/content/drive/MyDrive/data/gan1/dataset/archive"
SNAP = 4

# Build the command and run it
cmd = f"/usr/bin/python3 /content/stylegan2-ada-pytorch/train.py "\
  f"--snap {SNAP} --outdir {EXPERIMENTS} --data {DATA}"
!{cmd}




# for resume training
import os

# Modify these to suit your needs
EXPERIMENTS = "/content/drive/MyDrive/data/gan1/experiments"
NETWORK = "network-snapshot-000016.pkl"
RESUME = os.path.join(EXPERIMENTS, \
                "00009-archive-auto1-resumecustom", NETWORK)
DATA = "/content/drive/MyDrive/data/gan1/dataset/archive"
SNAP = 1

# Build the command and run it
cmd = f"/usr/bin/python3 /content/stylegan2-ada-pytorch/train.py "\
  f"--snap {SNAP} --resume {RESUME} --outdir {EXPERIMENTS} --data {DATA}"
!{cmd}

