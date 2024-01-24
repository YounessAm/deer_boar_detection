import os
import subprocess

HOME = os.getcwd()
HOME = os.path.dirname(HOME)
os.chdir(HOME)

# Cloning Groundin Dino directory and installing requirements
subprocess.run(["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"])
os.chdir(os.path.join(HOME, "GroundingDINO"))
subprocess.run(["pip", "install", "-q", "-e", "."])

# Define the config file
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

# Downloading Grounding Dino Weights
os.chdir(HOME)
os.makedirs(os.path.join(HOME, "weights"), exist_ok=True)
os.chdir(os.path.join(HOME, "weights"))
subprocess.run(["wget", "-q", "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"])

# Define the weight file
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

