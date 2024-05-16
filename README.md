# CTF-DDI
**CTF-DDI**:constrained tensor factorization for drug-drug interactions prediction

![61b79d17aba53ee954b6e48906bc18bd_4_Figure_1_1415366645](https://github.com/angelfacedac/CTF_DDI/assets/156782507/14219cc2-3676-4581-a00c-184a5864ed13)


# File description

- #### DS1: dataset1
- #### DS2: dataset2
- #### OUT: Save output file
- #### data.py: Read in the dataset and process it to obtain input data acceptable for the model
- #### experiments.py: The main function calls the model for training and testing evaluation
- #### method.py: Model code
- #### TDRC.py, TFAI.py: baseline code
- #### utils.py: Some tools involved


# Requirement

- #### python==3.9
- #### numpy
- #### tensorly
- #### scipy
- #### torch
- #### sklearn
- #### pandas
- #### matplotlib


# Usage

```
python experiments.py
```


