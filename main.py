
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random





#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#print(X_encoded.head(10))
#X_encoded.to_csv('pacy.txt', sep='\t')#, index=False)


if __name__ == "__main__":
    print('hello, world')

    data = pd.read_csv('car_step2.txt', delimiter='\t')

    print(data.head(10))

    X = data[['price_eur', 'year', 'km_age']]
    y = data['mark']
    encoded_brands = pd.get_dummies(y)
    table = pd.concat([X, encoded_brands], axis=1)

    #data=np.array([[-2.75,-4.5],[7.25,13.5],[4.25,8.5],[-8.75,-17.5]]) #w,m,m,w

    #all_y_true=np.array([1,0,0,1])#0-man, 1-women

    table = table.sample(frac=1).reset_index(drop=True)


    full_data = table[['year', 'km_age' ]].values
    full_y = table['price_eur'].values


    train_data = full_data[:970]
    train_y = full_y[:970]

    test_data = full_data[970:]
    test_y = full_y[970:]

    losses=[]

    #network=NeuralNetwork()
    #network.train(data, all_y_true)


    #for d, y in zip(test_data, test_y):
        #p= network.feedforward(d)*max_price
        #print(p,y*max_price)


    #w=np.array([0.25,-9.5])
    #m=np.array([12.55,3.2])

    #print('W', network.feedforward(w))
    #print('M', network.feedforward(m))

    #plt.plot(losses)
    #plt.xlim(left=0)
    #plt.ylim(bottom=0)
    #plt.show()

