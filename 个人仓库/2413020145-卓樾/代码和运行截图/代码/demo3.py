import os
import torch
import pandas as pd

os.makedirs(os.path.join("..","data"),exist_ok=True)
data_file = os.path.join("..","data","house_tiny.cvs")
with open(data_file,"w") as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

data = pd.read_csv(data_file)
print(data)

#插值法
inputs, outputs = data.iloc[:, 0:2], data.iloc[:,2]
inputs = inputs.fillna( inputs.mean(numeric_only=True) )
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
inputs = inputs.astype(int)
print(inputs)

#数值型dataframe转化为张量
x,y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x,y,sep="\n")