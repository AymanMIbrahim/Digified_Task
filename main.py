from Data_Creation import Generate_Data
from Data_preprocessing import Data_preprocessing
from Model_Creation import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if __name__ == "__main__":

    if not os.path.exists("./weights"):
        os.mkdir("./weights/")

    if not os.path.isfile("./Real_Names.pkl"):
        Generate_Data()

    if not os.path.isfile("./Dataset_Data.pkl"):
        Data_preprocessing()

    model = create_model()
    train(model,"./weights/",version="V1")

