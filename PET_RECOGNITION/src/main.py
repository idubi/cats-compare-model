from FeatureExtractot import FeatureExtractor 
from SiameseNetwork import SiameseNetwork
from TripletNetwork import  TripletNetwork
from DB.FAISSWrapper import FAISSWrapper
import numpy as np


siamese_model = SiameseNetwork("resnet50")
print("Siamese Network Ready ✅")
triplet_model = TripletNetwork("resnet50")
print("Triplet Network Ready ✅")
fais = FAISSWrapper()
# fais.add_pet(2, "Bella", "dog", np.random.rand(128).astype("float32"))


