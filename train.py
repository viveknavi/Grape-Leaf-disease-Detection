import os
import test
import numpy as np
import pandas as pd
import feature_extract

names = ['area','perimeter','physiological_length','physiological_width','aspect_ratio','rectangularity','circularity',
             'mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b','dissimilarity_l', 'contrast_l', 'homogeneity_l', 'energy_l', 'correlation_l',
             'dissimilarity_u', 'contrast_u', 'homogeneity_u', 'energy_u', 'correlation_u','class']
df = pd.DataFrame([], columns=names)
for root, dirs, files in os.walk("dataset", topdown=False):
    print(root)
    classLabel=0
    for file in files:
        
        filePath = os.path.join(root, file)
        vector=feature_extract.featureExtraction(filePath,0)
        print(vector)
        d = root.split('\\')
        print(d)
        vector.append(d[2])
        print(len(names),len(vector))
        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)

df.to_csv("grape_features.csv")
