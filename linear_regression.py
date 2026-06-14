import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("house_price_dataset_150.csv")
print(df.info())
df.describe()
x=df[["Area_sqft"]]
y=df["Price_INR"]
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
plt.scatter(x_train,y_train)
model=LinearRegression()
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)

predict_train=model.predict(x_train)
plt.scatter(x_train,y_train)
plt.plot(x_train,predict_train,color="red")
predictions=model.predict(x_test)
result=pd.DataFrame({"Actual":y_test,"Predicted":predictions})
print(result)
print(result.shape)
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mae=mean_absolute_error(y_test,predictions)
print(mae)
r2_score=r2_score(y_test,predictions)
print(r2_score)
mse=mean_squared_error(y_test,predictions)
print(mse)
area=float(input("Enter the Area:"))
print(f"House price for {area} has predicted price of {(model.predict([[area]]))}")

