from transformers import AutoTokenizer,BertModel
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from arabert import ArabertPreprocessor
import pickle

def Data_preprocessing():
    with open('Real_Names.pkl', 'rb') as f:
        Real_Names = pickle.load(f)
        Real_Names = Real_Names[0:10000]
    with open('Fake_Names.pkl', 'rb') as f:
        Fake_Names = pickle.load(f)
        Fake_Names = Fake_Names[0:10000]


    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-large-arabertv02")
    model = BertModel.from_pretrained("aubmindlab/bert-large-arabertv02")
    preprocess_model_name = "bert-base-arabertv2"
    arabert_prep = ArabertPreprocessor(model_name=preprocess_model_name)
    X = []
    Y = []

    for text in tqdm(Real_Names):

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
        Y.append(1)

    for text in tqdm(Fake_Names):

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
        Y.append(0)

    X = np.vstack(X)
    Y = np.vstack(Y)

    with open('Dataset_Data.pkl', 'wb') as f:
        pickle.dump(X, f)
        print("[INFO] Data prepocessed and saved succefully")

    with open('Dataset_Target.pkl', 'wb') as f:
        pickle.dump(Y, f)
        print("[INFO] Target data preprocessed and saved succefully")