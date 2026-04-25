import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

file=pd.read_csv("cleaned_data.csv")

file['datetime']=pd.to_datetime(file['datetime'])

file['hour']=file['datetime'].dt.hour+file['datetime'].dt.minute/60

file = file.sort_values('datetime')
file['demandH1']=file['demand'].shift(1).fillna(0)
file['demandH2'] = file['demand'].shift(2).fillna(0)
file['demandH3'] = file['demand'].shift(3).fillna(0)

file['hour_sin']=np.sin(2*np.pi*file['hour']/24)
file['hour_cos']=np.cos(2*np.pi*file['hour']/24)

file['minutes']=file['datetime'].dt.hour*60+file['datetime'].dt.minute
file['hour^2']=file['minutes']**2
file['hour^3']=file['minutes']**3

file['demand24HAgo']=file['demand'].shift(24).fillna(0)
file['demand7HAgo']=file['demand'].shift(24*7).fillna(0)

hourOneHotEncoding=pd.get_dummies(file['hour'],prefix='h').astype(int)
file=pd.concat([file,hourOneHotEncoding],axis=1)
hour_columns=[col for col in file.columns if col.startswith('h_')]

for h in hour_columns:
    file[f'{h}_weekend']=file[h]*file['is_weekend']

file['rush']=( (file['hour'] == 7) & (file['is_weekend'] == 0)).astype(int)
cols=[col for col in file.columns if col.endswith('_weekend')]
# print(file.head()) # print the beginning of the file
# print(file.info()) # Print name of columns

features=['minutes','is_weekend','event','rush','demandH1','demandH2','demandH3','hour_sin','hour_cos','demand24HAgo','demand7HAgo']+cols+hour_columns
x=file[features]
y=file['demand']



model=LinearRegression()
model.fit(x,y)

file['prediction_linear']=model.predict(x)
print(file[['datetime','demand','prediction_linear']].head(1000))


# TESTS EFFICAICTE MODELE
mean=mean_absolute_error(file['demand'],file['prediction_linear'])
print("Erreur moyenne :"+str(mean)+" passagers") 


# Save the results of the Linear Model
#file.to_csv("resultsLinearModel.csv",index=False)
