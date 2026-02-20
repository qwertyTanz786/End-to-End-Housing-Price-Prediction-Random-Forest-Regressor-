import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib
#read data
df=pd.read_csv(r"C:\Users\panch\Downloads\Training model_OWN\housing.csv")
#dropping the categorical column
df.drop('ocean_proximity',axis=1,inplace=True)
#filling in any missing values if any with median value of that column
df.fillna(df.median(),inplace=True)
#defining the feature and target variable
X=df.drop('median_house_value',axis=1)
Y=df['median_house_value']
#train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#model training
model=RandomForestRegressor()
#hypertuning the parameters
RandomForestRegressor(n_estimators=300,max_depth=20,min_samples_split=5,random_state=42)
#fit the model
model.fit(X_train,Y_train)
#prediction of the model and evaluation of the required metrics
pred=model.predict(X_test)
mse=mean_squared_error(Y_test,pred)
rmse=np.sqrt(mse)
r2=r2_score(Y_test,pred)
#prinitng the metrics
print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")
#saving the model 
joblib.dump(model,'boston_rf_model.pkl')