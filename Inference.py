from tensorflow import keras
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer,BertModel
from arabert import ArabertPreprocessor
from flask import Flask,request,jsonify
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
model = BertModel.from_pretrained("aubmindlab/bert-base-arabertv02")
preprocess_model_name = "bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=preprocess_model_name)
prediction_model = keras.models.load_model("./weights/BERT_BI-LSTM-V6_Digified_Text_Classification_24_0.87.hdf5")

def Preprocess_Input(text):
    global tokenizer,model,preprocess_model_name,arabert_prep
    text_list = [text]
    X = []
    for text in tqdm(text_list):
        text = arabert_prep.preprocess(text)
        inputs = tokenizer(text, return_tensors="pt")

        Pad = 3 - len(inputs["input_ids"][0])

        if Pad > 0:
            inputs["input_ids"] = torch.cat(
                (inputs["input_ids"], torch.tensor(np.array([[0] * Pad], dtype=np.int32))), 1)
            inputs["token_type_ids"] = torch.cat(
                (inputs["token_type_ids"], torch.tensor(np.array([[0] * Pad], dtype=np.int32))), 1)
            inputs["attention_mask"] = torch.cat(
                (inputs["attention_mask"], torch.tensor(np.array([[0] * Pad], dtype=np.int32))), 1)

        else:
            inputs["input_ids"] = (inputs["input_ids"][0][0:3]).unsqueeze(0)
            inputs["token_type_ids"] = (inputs["token_type_ids"][0][0:3]).unsqueeze(0)
            inputs["attention_mask"] = (inputs["attention_mask"][0][0:3]).unsqueeze(0)

        outputs = model(**inputs)
        outputs = outputs.last_hidden_state.detach().numpy()
        X.append(outputs)
        X = np.vstack(X)

        return X

app = Flask(__name__)

@app.route('/test',methods=["POST"])
def Inference():
    global prediction_model
    if request.method == "POST":
        req_json = request.json
        test_name = req_json["Name"]
        X = Preprocess_Input(text=test_name)
        res = prediction_model.predict(X)
        Index = np.argmax(res)
        Res_List = list(res[0])
        Real = str(round(Res_List[1]*100,2))
        Fake = str(round(Res_List[0]*100,2))

        return jsonify({"Real_Conf": Real, "Fake_Conf": Fake})


if __name__ == '__main__':
    app.run(debug=False,port=9090)
