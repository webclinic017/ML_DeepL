# intraday_ml_strategy.py

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy

from datetime import datetime, timedelta

from itertools import cycle
import os

from scipy.special import comb

class IntradayMachineLearningPredictionStrategy(AbstractStrategy):
    """
    Requires:
    tickers - The list of ticker symbols
    events_queue - A handle to the system events queue
    """
    def __init__(
        self, tickers, events_queue, 
        model_file, lags, model, return_win, percent_factor,
        salida, skip):
        
        self.tickers = tickers
        self.events_queue = events_queue
        self.lags = lags

        self.invested = "NONE"
        self.cur_prices = np.zeros(self.lags+1)

        self.cur_time = [None for i in range(self.lags+1)]
        self.delta = np.zeros(self.lags)
        
        self.cur_returns = np.zeros(self.lags)
        self.minutes = 0
        self.qty = 500
        self.modelpkl = joblib.load(model_file)
        
        self.RF = model
        self.percent_factor = percent_factor
        self.return_win = return_win
        #self.entry_price = 0.0
        self.conta = cycle(range(0,500))
        #self.botlimit = None
        #self.toplimit = None
        pred = self.modelpkl.predict(return_win)
        mean = np.mean(pred)
        sigma = np.std(pred)
#        self.botlimit = mean - self.percent_factor*sigma
#        self.toplimit = mean + self.percent_factor*sigma

        self.botlimit = mean - percent_factor
        self.toplimit = mean + percent_factor


#        self.botlimit = 0.0
#        self.toplimit = 0.0                
        
        self.salida = salida
        self.contador = 0
        
        self.current_directory = os.getcwd()
        self.skip = skip
        

    def _update_current_returns(self, event):
        """
        Updates the array of current returns "features"
        used by the machine learning model for prediction.
        """
        # Adjust the feature vector to move all lags by one
        # and then recalculate the returns
        for i, f in reversed(list(enumerate(self.cur_prices))):
#YOE(lags=2) cur_prices[lags+1]  min  Price  cur_returns[lags]     cur_time[lags+1]  delta[lags]
#            [Pr1 0     0]       0    Pr1    [0         0]         [T1     0    0]   [0     0]
#            [Pr2 Pr1   0]       1    Pr2    [0         0]         [T2    T1    0]   [0     0]
#            [Pr3 Pr2 Pr1]       2    Pr3    [0         0]         [T3    T2   T1]   [0     0]
#            [Pr4 Pr3 Pr2]       3    Pr4    [0         0]         [T4    T3   T2]   [0     0]  
#            [Pr5 Pr4 Pr3]       4    Pr5    [Pr5/Pr4-1 Pr4/Pr3-1] [T5    T4   T3]   [T5-T4 T4-T3]
            
            if i > 0:
                self.cur_prices[i] = self.cur_prices[i-1]
                self.cur_time[i] = self.cur_time[i-1]
            else:
                self.cur_prices[i] = event.close_price/float(
                    PriceParser.PRICE_MULTIPLIER
                )
                self.cur_time[i] = str(event.time)
                
        #if self.minutes > (self.lags + 1):  # original
        if self.minutes >= self.lags:
            for i in range(0, self.lags):
                self.cur_returns[i] = ((
                    self.cur_prices[i]/self.cur_prices[i+1]
                )-1.0)*100.0
        # YOE Checking continuity in the prices:       
                a = str(datetime.strptime(self.cur_time[i], "%Y-%m-%d %H:%M:%S") - 
                        datetime.strptime(self.cur_time[i+1], "%Y-%m-%d %H:%M:%S"))
#                print(self.cur_time[i], self.cur_time[i+1])
#                print(a)
                try:
                    a = datetime.strptime(a, "%H:%M:%S")
                except:
                    try:
                        a = datetime.strptime(a, "%d days, %H:%M:%S")
                    except:
                        a = datetime.strptime(a, "%d day, %H:%M:%S")
                        
                self.delta[i] = int(a.strftime("%M"))

    def calculate_signals(self, event):    
        if self.RF == "RFR":        
            """
            Calculate the intraday machine learning 
            prediction strategy.
            """
            if event.type == EventType.BAR:
                self._update_current_returns(event)
                self.minutes += 1
                # Allow enough time to pass to populate the 
                # returns feature vector                    
                    
                #if self.minutes > (self.lags + 2): # original
                if self.minutes > (self.lags - 1):
#                    chk = True                    
                    if all(x == self.skip for x in self.delta):
                        chk = True
                    else:
                        chk = False                                        
                        try:
                            f = open(self.current_directory + "/" + "Discontinuity.txt", "r")
                            f.close()
                            f = open(self.current_directory + "/" + "Discontinuity.txt", "a")
                            f.write(str(self.cur_time) + "\n")
                            f.close()
                        except IOError:
                            f = open(self.current_directory + "/" + "Discontinuity.txt", "w+")
                            f.write(str(self.cur_time) + "\n")
                            f.close()
                    if chk:
                        if self.poly >= 2:
                            cont = 0
                            for i in range(0, self.lags):
                                for j in range(i, self.lags):
                                    self.add_returns[cont] = self.cur_returns[i]*self.cur_returns[j]
                                    cont += 1 
                            if self.poly == 3:
                                for i in range(0, self.lags):
                                    for j in range(i, self.lags):
                                        for k in range(j, self.lags):
                                            self.add_returns[cont] = self.cur_returns[i]*self.cur_returns[
                                                    j]*self.cur_returns[k]
                                            cont += 1
                            cur_returns = np.concatenate((self.cur_returns, self.add_returns))
                        else:
                            cur_returns = self.cur_returns
                            
                        try:
                            f = open(self.current_directory +  "/" + "test.txt", "r")
                            f.close()
                            f = open(self.current_directory + "/" + "test.txt", "a")
                            f.write(str(event.time) + str(cur_returns) + "\n")
                            f.close
                        except IOError:
                            f = open(self.current_directory + "/" + "test.txt", "w+")
                            f.write(str(event.time) + str(cur_returns) + "\n")    
                            f.close


                        pred = self.modelpkl.predict(cur_returns.reshape((1, -1)))[0]
                        #print(event.time, pred)
                        try:
                            f = open(self.current_directory +  "/" + "pred_test.txt", "r")
                            f.close()
                            f = open(self.current_directory + "/" + "pred_test.txt", "a")
                            f.write(str(event.time) + "_" + str(pred) + "\n")
                            f.close
                        except IOError:
                            f = open(self.current_directory + "/" + "pred_test.txt", "w+")
                            f.write(str(event.time) + "_" + str(pred) + "\n")    
                            f.close


    #                    current_price = event.close_price/float(
    #                            PriceParser.PRICE_MULTIPLIER)
                        # Long only strategy
                        if self.invested == "NONE":
                            if pred > self.toplimit:
                                self.qty = int(round(30000/(event.close_price/float(
                                        PriceParser.PRICE_MULTIPLIER)),0)-1.0)
                                print("LONG: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "BOT", self.qty)
                                        )
                                self.invested = "LONG"
                                self.contador = 0
                                #self.entry_price = event.close_price/float(
                                #        PriceParser.PRICE_MULTIPLIER)
#                            elif pred < self.botlimit:
#                                self.qty = int(round(30000/(event.close_price/float(
#                                        PriceParser.PRICE_MULTIPLIER)),0)-1.0)
#                                print("SHORT: %s" % event.time)
#                                self.events_queue.put(
#                                        SignalEvent(self.tickers[0], "SLD", self.qty)
#                                        )
#                                self.invested = "SHORT"
#                                self.contador = 0
#                                #self.entry_price = event.close_price/float(
#                                #        PriceParser.PRICE_MULTIPLIER)
    
                        elif self.invested == "LONG":
                            self.contador += 1
    #                        if current_price > self.entry_price:
    #                            print("CLOSING LONG: %s" % event.time)
    #                            self.events_queue.put(
    #                                    SignalEvent(self.tickers[0], "SLD", self.qty)
    #                                    )
    #                            self.invested = "NONE"
                            #if pred < self.botlimit:
                            if self.contador == self.salida:
                                print("CLOSING LONG: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "SLD", self.qty)
                                        )
                                self.invested = "NONE"
#                        elif self.invested == "SHORT":
#                            self.contador += 1
#    #                        if current_price < self.entry_price:
#    #                            print("CLOSING LONG: %s" % event.time)
#    #                            self.events_queue.put(
#    #                                    SignalEvent(self.tickers[0], "BOT", self.qty)
#    #                                    )
#    #                            self.invested = "NONE"
#                            #if pred > self.toplimit:
#                            if self.contador == self.salida:
#                                print("CLOSING SHORT: %s" % event.time)
#                                self.events_queue.put(
#                                        SignalEvent(self.tickers[0], "BOT", self.qty)
#                                        )
#                                self.invested = "NONE"                        
        elif self.RF == "RFC":
            """
            Calculate the intraday machine learning 
            prediction strategy.
            """
            if event.type == EventType.BAR:
                self._update_current_returns(event)
                self.minutes += 1
                # Allow enough time to pass to populate the 
                # returns feature vector
                if self.minutes > (self.lags + 2):
#                    chk = True                    
                    if all(x == self.skip for x in self.delta):
                        chk = True
                    else:
                        chk = False                                        
                        try:
                            f = open(self.current_directory + "/" + "Discontinuity.txt", "r")
                            f.close()
                            f = open(self.current_directory + "/" + "Discontinuity.txt", "a")
                            f.write(str(self.cur_time) + "\n")
                            f.close()
                        except IOError:
                            f = open(self.current_directory + "/" + "Discontinuity.txt", "w+")
                            f.write(str(self.cur_time) + "\n")
                            f.close()
                    if chk:
                        pred = self.modelpkl.predict(self.cur_returns.reshape((1, -1)))[0]
                        if self.invested == "NONE":
                            if pred == 1:
                                self.qty = int(round(30000/(event.close_price/float(
                                        PriceParser.PRICE_MULTIPLIER)),0)-1.0)
                                print("LONG: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "BOT", self.qty)
                                        )
                                self.invested = "LONG"
                                self.contador = 0
                            elif pred == -1:
                                self.qty = int(round(30000/(event.close_price/float(
                                        PriceParser.PRICE_MULTIPLIER)),0)-1.0)
                                print("SHORT: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "SLD", self.qty)
                                        )
                                self.invested = "SHORT"
                                self.contador = 0
    
                        elif self.invested == "LONG":
                            self.contador += 1
                            if self.contador == self.salida:
#                            if pred == -1:
                                print("CLOSING LONG: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "SLD", self.qty)
                                        )
                                self.invested = "NONE"
                        elif self.invested == "SHORT":
                            self.contador += 1
                            if self.contador == self.salida:
#                            if pred == 1:
                                print("CLOSING SHORT: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "BOT", self.qty)
                                        )
                                self.invested = "NONE"                        

