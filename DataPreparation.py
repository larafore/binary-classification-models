import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn
from sklearn.model_selection import train_test_split

data = pd.read_csv("bank-full.csv", sep=",")

def create_unique_values(value_name):
    arr = []
    for value in data[value_name]:
        if value not in arr:
            arr.append(value)
    return arr

def create_key_value_pairs(arr):
    l = len(arr)
    mapping = {}
    for i in range(l):
        key = arr[i]
        value = i + 1
        mapping[key] = value

    return mapping
            
job_names = create_unique_values("job")
marital_names = create_unique_values("marital")
education_names = create_unique_values("education")
default_names = create_unique_values("default")
housing_names = create_unique_values("housing")
loan_names = create_unique_values("loan")
contact_names = create_unique_values("contact")
month_names = create_unique_values("month")
poutcomes_names = create_unique_values("poutcome")
y_names = create_unique_values("y")

job_dict = create_key_value_pairs(job_names)
marital_dict = create_key_value_pairs(marital_names)
education_dict = create_key_value_pairs(education_names)
default_dict = create_key_value_pairs(default_names)
housing_dict = create_key_value_pairs(housing_names)
loan_dict = create_key_value_pairs(loan_names)
contact_dict = create_key_value_pairs(contact_names)
month_dict = create_key_value_pairs(month_names)
poutcomes_dict = create_key_value_pairs(poutcomes_names)
y_dict = create_key_value_pairs(y_names)


data_file = open("bank-full.csv", "r")
new_file = open("bank-numbers.csv", "w")

data_list = list(data_file)

new_file.write(data_list[0])

for line in data_list[1:]:
    temp = []
    ll = line.split(",")
    temp.append(ll[0])
    temp.append(job_dict[ll[1]])
    temp.append(marital_dict[ll[2]])
    temp.append(education_dict[ll[3]])
    temp.append(default_dict[ll[4]])
    temp.append(ll[5])
    temp.append(housing_dict[ll[6]])
    temp.append(loan_dict[ll[7]])
    temp.append(contact_dict[ll[8]])
    temp.append(ll[9])
    temp.append(month_dict[ll[10]])
    temp.append(ll[11])
    temp.append(ll[12])
    temp.append(ll[13])
    temp.append(ll[14])
    temp.append(poutcomes_dict[ll[15]])

    # deleting new line
    sixteen_length = len(ll[16])
    ll[16] = ll[16][:sixteen_length-1]
    temp.append(y_dict[ll[16]])

    leng = len(temp)
    for i in range(leng):
        temp[i] = str(temp[i])    
        
    new_file.write(",".join(temp)+"\n")

new_file.close()

