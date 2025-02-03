import torchvision
from model.model import PlantDiseaseModel
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True,
	help="path to trained model model")
ap.add_argument("--img_path", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

