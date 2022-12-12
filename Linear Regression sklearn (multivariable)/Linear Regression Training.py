import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import pickle

def load_data():
    df=pd.read_csv('Dataset_KC_House\kc_house_data.csv')
    #check how many columns are their
    #print(len(df.columns))

    #Display all the columns and select the most relevent ones 
    print(df.columns)
    '''
    The Dataframe with parameters used. Adjust features based on regression score (more the greater and has relationship)
    df=df[["bedrooms","bathrooms","sqft_living","sqft_lot","floors",'waterfront','view','condition','grade','yr_built','yr_renovated','sqft_living15','sqft_lot15','price']]
    '''

    df=df[["bedrooms","sqft_living","sqft_lot",'condition','grade','yr_renovated','yr_built','floors','view','sqft_basement','sqft_above','waterfront','sqft_living15','sqft_lot15','bathrooms','lat','long','price']]
    df=df.dropna()
    
    #Display the updated dataframe
    print(df)
    return df

def train_linear_model(dataset):
    
    X=dataset.drop('price',True) #Get all columns except price as features
    Y=dataset['price']
    
    X=np.array(X[X.columns])
    Y=np.array(Y.to_list()).reshape(-1,1)
    
    #Y=np.array(dataset['Height']).reshape(-1,1)
    #X=np.array(dataset['Weight']).reshape(-1,1)
    reg=LinearRegression().fit(X,Y)
    print(reg.score(X,Y))
    return reg

if __name__=="__main__":
    dataset=load_data()

    model_trained=train_linear_model(dataset)
    
    #Saving model 
    pickle.dump(model_trained,open('linear_Regression.pkl','wb'))
