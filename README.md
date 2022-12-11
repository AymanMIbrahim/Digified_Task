# Digified_Task For Arabic Fake Name Detection
## There are 4 Main steps in this task
*  Data generation: Which is to generate real and fake data from the given dataset
*  Data preprocessing: Which is to apply feature extraction using arabert model and apply padding in order to make the input in a fixed shape
*  Model creation: Which is creating bi-directional lstm model for training
*  Inference and API creation: Which is a flask API for testing our model
## Model and Important files to Download:
* Weights: https://drive.google.com/file/d/10B3-HGvJu_kE1nsFQVUHnsPT6UtputtD/view?usp=share_link
* Data: https://drive.google.com/file/d/1VsJytXi6IqbUB0sXQ6G5sgasNLGQ-anm/view?usp=share_link
* Target: https://drive.google.com/file/d/13NqB5hyx7-jKasKUrZ9Y34R5X5goYOic/view?usp=share_link
* Real Name:https://drive.google.com/file/d/1l7qfmDEqGSJFwX9Zpkd6tuhRlLL3pNWL/view?usp=share_link
* Fake Name: https://drive.google.com/file/d/1rGKawhqVA-xf00knvYNPF4h87lwBEjKJ/view?usp=share_link
## Installing dependencies:
* pip install -r requirements.txt
## Training:
* open main.py
* import Data_Creation and run Generate_Data in order to generate data
* import Data_preprocessing and run Data_preprocessing in order to convert the input to feature tensor for each name in the dataset
* import model_creation.py and create model using create_model
* using the created model now you can train your model using train(created_model , path_to_weights_folder , version)
## Inference:
* Run Inference.py to make the server running 
* Use post man in order to send Post request to the code with json file like this
{
    "Name": "ايمن محمد ابراهيم"
}
* The Input Key should called "Name"
* The output should be something like this
{
    "Fake_Conf": "99.83",
    "Real_Conf": "0.17"
}
