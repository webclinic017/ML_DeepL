import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import os
import glob
import pickle

from keras.models import Sequential
from keras.layers import Dense

collect_data = True
if collect_data:

    std = []                 # premarket std/day (in percent of the opening)
    opening = []             # market opening/day
    maxi = []                # Max value in the 1st hour of market/day
    mini = []                # Min value in the 1st hour of market/day
    vol_price = []           # average pre-market volume*price/day
    entry_point = []         # Entry point (in percent of the opening)
    intraday_premarket = []
    intraday_market = []

  #
  # Data between 4:00 am and 9:30 am, used to train the model
  # Data between 9:30 am and 10:00 am used to clasify the data
  #
    
    cuatroam = datetime(1900, 1, 1, 4, 0).time()
    nuevemedia = datetime(1900, 1, 1, 9, 30).time()
    diesmedia = datetime(1900, 1, 1, 10, 30).time()
    cuatropm = datetime(1900, 1, 1, 16, 0).time()
    ocho = datetime(1900, 1, 1, 20, 0).time()
    
    dias = 0
    
    current_directory = os.getcwd()
    filenames = glob.glob(current_directory)

  #
  # Reading data from csv files. This data will be cleaned and properly modified,
  # then saved i a new general csv file used to train the model.
  #    
    for filename in filenames:
        print("Colecting Data for:", filename)
        
        ts = pd.read_csv(filename, names=["Timestamp", "Open", "High", "Low",
                                 "Close", "Volume"], index_col="Timestamp", 
                                parse_dates=True)
        #ts.drop(["Open", "Low", "High", "Volume"], axis=1, inplace=True)
        
        print("Colecting Data, Done ...")
            
        temp_premarket = []
        temp_market = []
        close_volume = []
        open_high_low = []
        primero = ts.index[0]
        interv = 1
        cancelar = False
        j = 0
        if primero.time() < diesmedia:
            j = 1
            while ts.index[j].time() <= diesmedia:
                print("First before 10:30 am, waiting")
                j += 1
            primero = ts.index[j]

  #
  # Orginizing the data
  #    
        print("Orginizing Data for ", filename)
        
        for i in range(j, len(ts["Close"])):
            if (ts.index[i].date() - primero.date()).days == 0:
                if interv == 1:
                    if ts.index[i].time() >= cuatropm and ts.index[i].time() < ocho:
                        temp_premarket.append([ts["Close"].iloc[i], ts["Volume"].iloc[i], ts["High"].iloc[i], 
                                     ts["Low"].iloc[i], ts["Open"].iloc[i]])
                        interv = 1
                    if ts.index[i].time() > nuevemedia and ts.index[i].time() < diesmedia:
                        temp_market.append([ts["Open"].iloc[i], ts["High"].iloc[i], ts["Low"].iloc[i],
                                       ts["Close"].iloc[i], ts["Volume"].iloc[i]])
                if interv == 2:
                    if ts.index[i].time() >= cuatroam and ts.index[i].time() < nuevemedia:
                        temp_premarket.append([ts["Close"].iloc[i], ts["Volume"].iloc[i], ts["High"].iloc[i], 
                                     ts["Low"].iloc[i], ts["Open"].iloc[i]])
                        interv = 2
                    if ts.index[i].time() >= nuevemedia:
                        temp_market.append([ts["Open"].iloc[i], ts["High"].iloc[i], ts["Low"].iloc[i],
                                       ts["Close"].iloc[i], ts["Volume"].iloc[i]])
                        interv = 1
                if ts.index[i].time() == diesmedia:
  #
  # Filling empty spaces
  #    
                    if not cancelar:
                        open_high_low.append(temp_market)
                        temp_market = []
                        close_volume.append(temp_premarket)
                        temp_premarket = []
                    if cancelar:
                        open_high_low.append([[0.0,0.0,0.0]])
                        temp_market = []
                        close_volume.append([[0.0,0.0,0.0,0.0]])
                        temp_premarket = []
                    cancelar = False
                        
  # Day change:
            if (ts.index[i].date() - primero.date()).days != 0:
                primero = ts.index[i]
                if ts.index[i].time() >= cuatroam and ts.index[i].time() < nuevemedia:
                    interv = 2
                    temp_premarket.append([ts["Close"].iloc[i], ts["Volume"].iloc[i], ts["High"].iloc[i], 
                                 ts["Low"].iloc[i], ts["Open"].iloc[i]])
                    cancelar = False
  # If no data between 4.00 am and 9:20am:     
                elif ts.index[i].time() == nuevemedia:
                    interv = 1
                    temp_market.append([ts["Open"].iloc[i], ts["High"].iloc[i], ts["Low"].iloc[i],
                                   ts["Close"].iloc[i], ts["Volume"].iloc[i]])
                    cancelar = False
  # If no data at 9:30am:
                elif ts.index[i].time() > nuevemedia and ts.index[i].time() < diesmedia:
                    interv = 1
                    temp_market.append([ts["Open"].iloc[i], ts["High"].iloc[i], ts["Low"].iloc[i],
                                   ts["Close"].iloc[i], ts["Volume"].iloc[i]])
                    cancelar = True
                elif ts.index[i].time() > diesmedia:
                    interv = 1                
                    open_high_low.append([[0.0,0.0,0.0]])
                    temp_market = []
                    close_volume.append([[0.0,0.0,0.0,0.0]])
                    temp_premarket = []                
    
    
        
        if len(close_volume) != len(open_high_low):        
            print("The amount of pre-market data is different to the amount of market data")
            print(len(close_volume), len(open_high_low))
    
        dias += len(open_high_low)
        
        print("Orginizing Data. Done ...")
                
        print("Computing", filename)
        for i in range(len(open_high_low)):
            premarket_per_day = np.array(close_volume[i])
            market_per_day = np.array(open_high_low[i])
    
            abre = market_per_day[0,0]

  # Days with no pre-market data do not enter in the array

            if abre != 0:
                temp_premarket = []
                temp_market = []
                for j in range(len(premarket_per_day)):
                    temp_premarket.append([premarket_per_day[j][4], premarket_per_day[j][2],
                                               premarket_per_day[j][3], premarket_per_day[j][0], 
                                               premarket_per_day[j][1]])
                for j in range(len(market_per_day)):
                    temp_market.append(market_per_day[j])
            
                intraday_premarket.append(temp_premarket)
                intraday_market.append(temp_market)
            
                opening.append(abre)
                standard = np.std(premarket_per_day[:,0])
                std.append(standard*100/abre)
                media = np.mean(premarket_per_day[:,0])
                entrada = (abre - media)*100/abre
                entry_point.append(entrada)
                vol_price.append(np.sum(((premarket_per_day[:,2]+premarket_per_day[:,3])/2)*premarket_per_day[:,1]))
                mini.append(np.min(market_per_day[:,2]))
                maxi.append(np.max(market_per_day[:,1]))
                
#            else:
#                std.append(0.0)
#                entry_point.append(0.0)                
#                vol_price.append(0.0)
#                mini.append(0.0)
#                maxi.append(0.0)
    
    data = pd.DataFrame()
    data['volume_price'] = vol_price
    data['volatility'] = std
    data['entry'] = entry_point
    data['opening'] = opening
    data['mini'] = mini
    data['maxi'] = maxi

  #
  # Saving the clean and orginized data
  #
    data.to_csv(current_directory + "/"+ "data_saved" + ".csv", sep = ",", index = False)    

  #
  # Collecting data for training
  #

current_directory = os.getcwd()
datos = pd.read_csv(current_directory + "/"+ "data_saved" + ".csv", index_col = False)

entry_point = []
std = []
vol_price = []
opening = []
mini = []
maxi = []

dias = len(datos['volume_price'])
for i in range(dias):
    entry_point.append(datos['entry'][i])
    std.append(datos['volatility'][i])
    vol_price.append(datos['volume_price'][i])
    opening.append(datos['opening'][i])
    mini.append(datos['mini'][i])
    maxi.append(datos['maxi'][i])

X=[]
Y=[]    
cont_neg = 0
cont_pos = 0
for i in range(dias):
    if opening[i] != 0:
        X.append([entry_point[i], std[i], vol_price[i]])
        if opening[i] + opening[i]*0.16/100 <= maxi[i]:
#        if opening[i] - opening[i]*0.16/100 >= mini[i]:
            Y.append(1)
            cont_pos += 1
        else:
            Y.append(0)
            cont_neg += 1
    else:
        print("Yoe, zero value !!!!!!!!!!!!!!!!!!!!!!!!!!")
print("Good trades", cont_pos)
print("Bad trades", cont_neg)

hit_rate_train = []
hit_rate_test = []
yoe_prob = []
num_senal = []

X = np.array(X)
Y = np.array(Y)

intraday_premarket = np.array(intraday_premarket)
intraday_market = np.array(intraday_market)

  #
  # Using Kfold to avoid overfitting
  #

kf = KFold(n_splits = 2, shuffle = True, random_state = 17)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
        
            
quinto = int(len(X)/5)
X_train = X[:len(X) - quinto]
Y_train = Y[:len(Y) - quinto]

X_test = X[-quinto:]
Y_test = Y[-quinto:]
                    

  #########################
  #  Deep Learning
  # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
  #########################

X_train = np.array(X_train)
X_test = np.array(X_test)

  #
  # 4 layers with 32 nodes with relu activation
  # 1 layer with 8 nodes with relu activation
  # 1 sigmoid layer to clasify the data
  #

model = Sequential()
model.add(Dense(32, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  # fitting

model.fit(X_train, Y_train, epochs=150, batch_size=32, verbose=2)

  # evaluating the model

scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print("Computing test evaluation ...")
score_test = model.evaluate(X_test, Y_test, verbose=2)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])

#y_pred = model.predict_proba(Y_test)

print("Case 1")
print("")

good = 0
std016 = 0
for i in range(len(open_high_low)):
    if std[i] > 0.16*2:
        std016 += 1

    if opening[i] < mean[i] and opening[i] + opening[i]*0.16/100 <= maxi[i] and std[i] > 0.16*2:
        good += 1
    if opening[i] > mean[i] and opening[i] - opening[i]*0.16/100 >= mini[i] and std[i] > 0.16*2:
        good += 1
        
print(good/std016)
print(len(open_high_low))
print(std016)



temp2 = np.array(open_high_low[1])
print(temp2[0,0])
vol_price.append(np.sum(((premarket[:,2]+premarket[:,3])/2)*premarket[:,1]))
print(premarket[:,2], premarket[:,3], premarket[:,1])
print(((premarket[:,2]+premarket[:,3])/2)*premarket[:,1])

print(temp[:,0])
print(mean[0])
print(std[0])
print(opening[0])
print(mini[0])
print(maxi[0])
print(volume[0])




## Get Adjusted Closing Prices for Facebook, Tesla and Amazon between 2016-2017
#fb = get_adj_close('fb', '1/2/2016', '31/12/2017')
#tesla = get_adj_close('tsla', '1/2/2016', '31/12/2017')
#amazon = get_adj_close('amzn', '1/2/2016', '31/12/2017')
#
## Calculate 30 Day Moving Average, Std Deviation, Upper Band and Lower Band
#for item in (fb, tesla, amazon):
#    item['30 Day MA'] = item['Adj Close'].rolling(window=20).mean()
#    item['30 Day STD'] = item['Adj Close'].rolling(window=20).std()
#    item['Upper Band'] = item['30 Day MA'] + (item['30 Day STD'] * 2)
#    item['Lower Band'] = item['30 Day MA'] - (item['30 Day STD'] * 2)
#
## Simple 30 Day Bollinger Band for Facebook (2016-2017)
#fb[['Adj Close', '30 Day MA', 'Upper Band', 'Lower Band']].plot(figsize=(12,6))
#plt.title('30 Day Bollinger Band for Facebook')
#plt.ylabel('Price (USD)')
#plt.show()
