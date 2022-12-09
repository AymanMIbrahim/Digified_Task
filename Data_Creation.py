import pandas as pd
from tqdm import tqdm
from random import randint
from random import shuffle
import pickle

def Fake_it(Name):
    ListOfChars  = ["ا","أ","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ص","ض","ط","ع",
                    "غ","ف","ق","ك","ل","م","ن","ه","و","ي","ء","ئ","ة","ؤ","ﻷ","لا"]

    NewName = Name + ListOfChars[randint(0,len(ListOfChars)-1)]
    return NewName

def Generate_Data():
    Female_Names = []
    Male_Names = []
    df = pd.read_csv("Arabic_names.csv")
    for i in tqdm(range(len(df["Name"]))):
        if df["Gender"][i] == "F":
            Female_Names.append(df["Name"][i])
        else:
            Male_Names.append(df["Name"][i])

    Real_Names = []
    for i in tqdm(range(len(Female_Names))):
        for j in range(0,len(Male_Names)-1,2):
            Name = Female_Names[i]+" "+Male_Names[j]+" "+Male_Names[j+1]
            Real_Names.append(Name)

    for i in tqdm(range(len(Male_Names))):
        for j in range(0,len(Male_Names)-1,2):
            Name = Male_Names[i]+" "+Male_Names[j]+" "+Male_Names[j+1]
            Real_Names.append(Name)

    shuffle(Real_Names)
    Fake_Names = []

    for Name in Real_Names:
        ListOfName = Name.split()
        F_Name = ""
        for N in ListOfName:
            Fake = Fake_it(N)
            F_Name += Fake+" "
        Fake_Names.append(F_Name)

    shuffle(Fake_Names)

    print("Sample of Real Names: ",Real_Names[:20])
    print("Sample of Fake Names: ", Fake_Names[:20])
    with open('Real_Names.pkl', 'wb') as f:
        pickle.dump(Real_Names, f)
        print("[INFO] Real names generated and saved succefully")

    with open('Fake_Names.pkl', 'wb') as f:
        pickle.dump(Fake_Names, f)
        print("[INFO] Fake names generated and saved succefully")




