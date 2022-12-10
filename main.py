from Data_Creation import Generate_Data
from Data_preprocessing import Data_preprocessing
from Model_Creation import *
from Inference import Inference

if __name__ == "__main__":
    #Generate_Data()
    #Data_preprocessing()
    #model = create_model()
    #train(model,"./weights/",version="V1")

    text = "أيمن محمد ابراهيم"
    Inference("weights/BERT_BI-LSTM-V1_Digified_Text_Classification_70_0.71.hdf5",text)

