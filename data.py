import csv
import os.path as osp
import numpy as np
import pandas as pd


class GetData(object):

    def __init__(self, drug_num=807):
        super().__init__()
        self.drug_num = drug_num
        self.S1, self.S2, self.X, self.CYP, self.NCYP = self.__get__DS3__data__()
        # self.X = self.CYP
        # self.S1, self.S2, self.X = self.__get__DS1__data__()

    def __get__DS3__data__(self):
        s1_type_name = [
            "chemicalSimilarityMat",
            "ligandSimilarityMat",
            "SideEffectSimilarityMat",
            "ATCSimilarityMat"
        ]
        s2_type_name = [
            "seqSimilarityMat",
            "GOSimilarityMat",
            "distSimilarityMat"
        ]

        s1_all = []
        s2_all = []

        for name in s1_type_name:
            df = pd.read_csv("DS3/" + name + ".csv")
            mat = df.values
            s1_all.append(mat)

        for name in s2_type_name:
            df = pd.read_csv("DS3/" + name + ".csv")
            mat = df.values
            s2_all.append(mat)

        s1 = np.mean(np.stack(s1_all), axis=0)
        s2 = np.mean(np.stack(s2_all), axis=0)

        x_all = []
        for name in ["CYPInteractionMat", "NCYPInteractionMat"]:
            df = pd.read_csv("DS3/" + name + ".csv")
            mat = df.values
            x_all.append(mat)

        x = np.dstack(x_all)

        CYP = pd.read_csv('DS3/CYPInteractionMat.csv').values
        NCYP = pd.read_csv('DS3/NCYPInteractionMat.csv').values
        CYP = np.reshape(CYP, (807, 807, 1))
        NCYP = np.reshape(NCYP, (807, 807, 1))

        return s1, s2, x, CYP, NCYP

    def __get__DS1__data__(self):
        s1_type_name = ["chem_Jacarrd_sim",
                        "sideeffect_Jacarrd_sim",
                        "offsideeffect_Jacarrd_sim"
                        ]
        s2_type_name = ["target_Jacarrd_sim",
                        "transporter_Jacarrd_sim",
                        "enzyme_Jacarrd_sim",
                        "pathway_Jacarrd_sim",
                        "indication_Jacarrd_sim"
                        ]

        s1_all = []
        s2_all = []

        for name in s1_type_name:
            df = pd.read_csv("DS1/" + name + ".csv", header=None)
            mat = df.values
            s1_all.append(mat)

        for name in s2_type_name:
            df = pd.read_csv("DS1/" + name + ".csv", header=None)
            mat = df.values
            s2_all.append(mat)

        s1 = np.mean(np.stack(s1_all), axis=0)
        s2 = np.mean(np.stack(s2_all), axis=0)

        x = pd.read_csv('DS1/drug_drug_matrix.csv', header=None).values
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        return s1, s2, x
