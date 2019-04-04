import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import os
import glob
import pickle

from qstrader import settings
from qstrader.strategy.base import AbstractStrategy
from qstrader.event import SignalEvent, EventType
from qstrader.compat import queue
from qstrader.trading_session import TradingSession
from qstrader.price_handler.opening_intraday_csv_bar import Opening_intraday_csv_bar
from Opening_strategy import Opening_strategy

#from keras.models import Sequential
#from keras.layers import Dense
def Opening():
    collect_data = True
    if collect_data:
    
        std = []     # std del premarket por dia (en porciento del opening)
        opening = []   # opening del market por dia
        maxi = []    # maximo de 1h de market por dia
        mini = []    # minimo de 1h de market por dia
        vol_price = []   # average volume*price del premarket por dia
        entry_point = []  #punto de entrada (en porciento del opening)
        intraday_premarket = []
        intraday_market = []
        
        cuatroam = datetime(1900, 1, 1, 4, 0).time()
        nuevemedia = datetime(1900, 1, 1, 9, 30).time()
        diesmedia = datetime(1900, 1, 1, 10, 30).time()
        cuatropm = datetime(1900, 1, 1, 16, 0).time()
        ocho = datetime(1900, 1, 1, 20, 0).time()
        
        dias = 0
        
        current_directory = os.getcwd()
        filenames = glob.glob(current_directory + "/*.csv")
        
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
                    #print("Primero antes de las 10:30 am, waiting")
                    j += 1
                primero = ts.index[j]
        
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
                            
        # Cambio de dia:
                if (ts.index[i].date() - primero.date()).days != 0:
                    primero = ts.index[i]
                    if ts.index[i].time() >= cuatroam and ts.index[i].time() < nuevemedia:
                        interv = 2
                        temp_premarket.append([ts["Close"].iloc[i], ts["Volume"].iloc[i], ts["High"].iloc[i], 
                                     ts["Low"].iloc[i], ts["Open"].iloc[i]])
                        cancelar = False
        # si no hay data entre 4am y 9:29:     
                    elif ts.index[i].time() == nuevemedia:
                        interv = 1
                        temp_market.append([ts["Open"].iloc[i], ts["High"].iloc[i], ts["Low"].iloc[i],
                                       ts["Close"].iloc[i], ts["Volume"].iloc[i]])
                        cancelar = False
        # si no hay 9:30:
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
                print("La cantidad de premarket data es diferente a la cantidad de marketdata")
                print(len(close_volume), len(open_high_low))
        
            dias += len(open_high_low)
            
            print("Orginizing Data. Done ...")
        
            #print(open_high_low)
                
            print("Computing", filename)
            for i in range(len(open_high_low)):
                premarket_per_day = np.array(close_volume[i])
                market_per_day = np.array(open_high_low[i])
        
                abre = market_per_day[0,0]            
                if abre != 0:  # los dias que no tiene premarket no entran en los array
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
        
        data.to_csv(current_directory + "/"+ "data_saved" + ".csv", sep = ",", index = False)
        
        f = open(current_directory + "/"+ "intraday_premarket" + ".pkl", 'wb')
        pickle.dump(intraday_premarket, f)
        f.close
        f = open(current_directory + "/"+ "intraday_market" + ".pkl", 'wb')
        pickle.dump(intraday_market, f)
        f.close
        print("pkls written")        
    
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
            print("Yoe, un zero!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Numero de good trades", cont_pos)
    print("Numero de bad trades", cont_neg)
    
    hit_rate_train = []
    hit_rate_test = []
    yoe_prob = []
    num_senal = []
    
    f = open(current_directory + "/"+ "intraday_premarket" + ".pkl", 'rb')
    intraday_premarket = pickle.load(f)
    f.close
    f = open(current_directory + "/"+ "intraday_market" + ".pkl", 'rb')
    intraday_market = pickle.load(f)
    f.close
    
    print("pkls loaded")
    
    X = np.array(X)
    Y = np.array(Y)
    
    intraday_premarket = np.array(intraday_premarket)
    intraday_market = np.array(intraday_market)
    
    kf = KFold(n_splits = 5, shuffle = True, random_state = 17)
    print("starting kfold")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        Y_backtest_premarket = intraday_premarket[test_index]
        Y_backtest_market = intraday_market[test_index]
        
        Y_backtest = pd.DataFrame()
        temp_tiempo = datetime(2000, 1, 1, 4, 0, 0)
        Date = []
        Open = []
        High = []
        Low = []
        Close = []
        Volume = []
        for i in range(len(Y_backtest_premarket)):
            for j in range(len(Y_backtest_premarket[i])):
                Date.append(temp_tiempo)
                Open.append(Y_backtest_premarket[i][j][0])
                High.append(Y_backtest_premarket[i][j][1])
                Low.append(Y_backtest_premarket[i][j][2])
                Close.append(Y_backtest_premarket[i][j][3])
                Volume.append(Y_backtest_premarket[i][j][4])
                temp_tiempo = temp_tiempo + timedelta(seconds=30)
            temp_tiempo = datetime.combine(temp_tiempo.date(), datetime(2000, 1, 1, 9, 30, 0).time()) 
            for k in range(len(Y_backtest_market[i])):
                Date.append(temp_tiempo)
                Open.append(Y_backtest_market[i][k][0])
                High.append(Y_backtest_market[i][k][1])
                Low.append(Y_backtest_market[i][k][2])
                Close.append(Y_backtest_market[i][k][3])
                Volume.append(Y_backtest_market[i][k][4])
                temp_tiempo = temp_tiempo + timedelta(minutes=1)
            temp_tiempo = datetime(2000, 1, 1, 4, 0, 0) + timedelta(days=i+1)
        Y_backtest["Open"] = Open
        Y_backtest["High"] = High
        Y_backtest["Low"] = Low
        Y_backtest["Close"] = Close
        Y_backtest["Volume"] = Volume
        Y_backtest.index = Date # list(Y_backtest["Date"])
    #    Y_backtest.to_csv(current_directory + "/"+ "Y_backtest" + ".csv", sep = ",", index = False)
                
    #quinto = int(len(X)/5)
    #X_train = X[:len(X) - quinto]
    #Y_train = Y[:len(Y) - quinto]
    #
    #X_test = X[-quinto:]
    #Y_test = Y[-quinto:]
                        
        model = RandomForestClassifier(
                n_estimators=1000,  # Optimized for leaf 200
                n_jobs=2,
                random_state=75,
                #min_samples_split=self.min_samples_split,
                min_samples_leaf = 100,   # Optimized for y_pred_test[i][1] > 0.8 and tress 500
                oob_score = False
                #max_depth = self.max_depth,
                #max_features= "sqrt"
                )
        print("Fitting classifier model...")
        model.fit(X_train, Y_train)
    
        print("Confusion matrix: ")
        print(confusion_matrix(model.predict(X_test), Y_test))
        print("")
        print("Feature Importances: [Entry, Std, Volume]: ", model.feature_importances_)
    
        hit_rate_train.append(model.score(X_train, Y_train))
        print("Hit-Rate train: ", model.score(X_train, Y_train))
        hit_rate_test.append(model.score(X_test, Y_test))
        print("Hit-Rate test: ", model.score(X_test, Y_test))
    
    
        good = 0
        total = 0
        #y_pred_test = model.predict(X_test)
        y_pred_test = model.predict_proba(X_test)
        for i in range(len(X_test)):
            if y_pred_test[i][1] > 0.9:
                if Y_test[i] == 1:
                    good += 1
                    total += 1
                elif Y_test[i] == 0:
                    total += 1
        if total != 0:
            yoe_prob.append(good/total)
            num_senal.append(total)
            print("Yoe Prob: ", good/total)
            print("Total, Num_senhales: ", len(X_test), total)
            print("")
            print("###############################################")
            print("")
        else:
            print("")
            print(" There is no probability higher than 0.9")
            print("")
            print("###############################################")
            print("")
            
        # Configuration data
        testing = False
        config = settings.from_file(
                settings.DEFAULT_CONFIG_FILENAME, testing
                )
        tickers = ["ALL"]
        filename = None
        
        run(config, testing, tickers, filename, model, Y_backtest)
        
            
    
    print("Average Hit-Rate train: ", sum(hit_rate_train)/len(hit_rate_train))
    print("")
    print("Average Hit-Rate test: ", sum(hit_rate_test)/len(hit_rate_test))
    #oob_score = model.oob_score_
    #print("oob_score: %s\n" %oob_score)
    
    
    print("Average Yoe Prob: ", sum(yoe_prob)/len(yoe_prob))
    print("Average Total, Num_senhales: ", len(X_test), sum(num_senal)/len(num_senal))
    #print(model.predict_proba(X_test))


def run(config, testing, tickers, foldername, model, Y_backtest):
    # Backtest information
    title = ['Opening Strategy backtest %s' % tickers[0]]
    initial_equity = 90000.0
    start_date = datetime(2000, 1, 1)
    end_date = None
    #end_date = datetime(2014, 1, 1)

    # Use the Buy and Hold Strategy
    events_queue = None
    strategy = None
    csv_bar = None
    backtest = None
    results = None
    
    events_queue = queue.Queue()
    strategy = Opening_strategy(tickers[0], events_queue, model)
    csv_bar = Opening_intraday_csv_bar(Y_backtest, events_queue, ["ALL"],
                                       start_date, end_date)

    # Set up the backtest
    backtest = TradingSession(
        config, strategy, tickers,
        initial_equity, start_date, end_date,
        events_queue, price_handler=csv_bar,
        title=title
    )
    results = backtest.start_trading(testing=testing, foldername=foldername)
    return results


if __name__ == "__main__":
    # Configuration data
    testing = False
    config = settings.from_file(
        settings.DEFAULT_CONFIG_FILENAME, testing
    )
    tickers = ["ALL"]
    current_directory = os.getcwd()
    foldername = current_directory   # esto es para correr en el cluster
    #foldername = None
    
    Opening()
#    run(config, testing, tickers, filename)


#########################
#  Deep Learning
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#########################

#X_train = np.array(X_train)
#X_test = np.array(X_test)
#
#model = Sequential()
#model.add(Dense(32, input_dim=3, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, Y_train, epochs=150, batch_size=32, verbose=2)
#
## evaluate the model
#scores = model.evaluate(X_train, Y_train)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
#
#print("Computing test evaluation ...")
#score_test = model.evaluate(X_test, Y_test, verbose=2)
#print('Test loss:', score_test[0])
#print('Test accuracy:', score_test[1])
#
##y_pred = model.predict_proba(Y_test)

#print("Case 1")
#print("")
#
#good = 0
#std016 = 0
#for i in range(len(open_high_low)):
#    if std[i] > 0.16*2:
#        std016 += 1
#
#    if opening[i] < mean[i] and opening[i] + opening[i]*0.16/100 <= maxi[i] and std[i] > 0.16*2:
#        good += 1
#    if opening[i] > mean[i] and opening[i] - opening[i]*0.16/100 >= mini[i] and std[i] > 0.16*2:
#        good += 1
#        
#print(good/std016)
#print(len(open_high_low))
#print(std016)



#temp2 = np.array(open_high_low[1])
#print(temp2[0,0])

#vol_price.append(np.sum(((premarket[:,2]+premarket[:,3])/2)*premarket[:,1]))
#print(premarket[:,2], premarket[:,3], premarket[:,1])
#print(((premarket[:,2]+premarket[:,3])/2)*premarket[:,1])

#print(temp[:,0])
#print(mean[0])
#print(std[0])
#print(opening[0])
#print(mini[0])
#print(maxi[0])
#print(volume[0])

