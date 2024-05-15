import pandas as pd
import numpy as np

for path in ["ATCSimilarityMat",
             "chemicalSimilarityMat",
             "distSimilarityMat",
             "GOSimilarityMat",
             "ligandSimilarityMat",
             "seqSimilarityMat",
             "SideEffectSimilarityMat"]:
    df = pd.read_table(path, sep='\t', header=None)
    df.to_csv(path + ".csv", index=False)

df = pd.read_table("CYPInteractionMat.txt", sep='\t', header=None, encoding="utf-16")
df.to_csv("CYPInteractionMat.csv", index=False)

df = pd.read_table("NCYPInteractionMat.txt", sep='\t', header=None, encoding="utf-16")
df.to_csv("NCYPInteractionMat.csv", index=False)


if __name__ == '__main__':
    x_all = []
    for name in ["CYPInteractionMat", "NCYPInteractionMat"]:
        df = pd.read_csv(name + ".csv")
        mat = df.values
        x_all.append(mat)

    x = np.dstack(x_all)

    index_matrix = np.array(np.where(x == 1))

    print(index_matrix)

    np.random.shuffle(index_matrix.T)

    print(index_matrix)

