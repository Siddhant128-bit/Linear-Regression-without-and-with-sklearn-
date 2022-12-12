import pickle

if __name__=='__main__':
    model=pickle.load(open('linear_Regression.pkl','rb'))
    input_features=["bedrooms","sqft_living","sqft_lot",'condition','grade','yr_renovated','yr_built','floors','view','sqft_basement','sqft_above']
    inputs=[]
    for i in input_features: 
        inputs.append(float(input('Enter value for '+i+': ')))
    inputs=[inputs]
    print(model.predict(inputs))
