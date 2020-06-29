# intraday_ml_strategy.py
# Predicting using the previously trained pkl models

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy

from itertools import cycle

class IntradayMachineLearningPredictionStrategy(AbstractStrategy):
  """
  Requires:
  tickers - The list of ticker symbols
  events_queue - A handle to the system events queue
  """
    def __init__(
        self, tickers, events_queue, 
        model_pickle_file, lags, model, return_win, percent_factor,
        salida):
        
        self.tickers = tickers
        self.events_queue = events_queue
        self.lags = lags

        self.invested = "NONE"
        self.cur_prices = np.zeros(self.lags+1)
        self.cur_returns = np.zeros(self.lags)
        self.minutes = 0
        self.qty = 500
        self.modelpkl = joblib.load(model_pickle_file)
        
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
        self.botlimit = mean - self.percent_factor*sigma
        self.toplimit = mean + self.percent_factor*sigma
        
        self.salida = salida
        self.contador = 0

    def _update_current_returns(self, event):
  """
  Updates the array of current returns "features"
  used by the machine learning model for prediction.
  Adjust the feature vector to move all lags by one
  and then recalculate the returns
  """
        for i, f in reversed(list(enumerate(self.cur_prices))):
            if i > 0:
                self.cur_prices[i] = self.cur_prices[i-1]
            else:
                self.cur_prices[i] = event.close_price/float(
                    PriceParser.PRICE_MULTIPLIER
                )
        if self.minutes > (self.lags + 1):
            for i in range(0, self.lags):
                self.cur_returns[i] = ((
                    self.cur_prices[i]/self.cur_prices[i+1]
                )-1.0)*100.0

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
                if self.minutes > (self.lags + 2):
                    
#                    self.return_win.append(self.cur_returns)
#                    if next(self.conta) == 0:
#                        train_X = np.array([i for i in self.return_win])
#                        pred_win = self.modelpkl.predict(train_X)
#                        mean = np.mean(pred_win)
#                        sigma = np.std(pred_win)
#                        self.botlimit = mean - self.percent_factor*sigma
#                        self.toplimit = mean + self.percent_factor*sigma

                    pred = self.modelpkl.predict(self.cur_returns.reshape((1, -1)))[0]
                    current_price = event.close_price/float(
                            PriceParser.PRICE_MULTIPLIER)
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
                        elif pred < self.botlimit:
                            self.qty = int(round(30000/(event.close_price/float(
                                    PriceParser.PRICE_MULTIPLIER)),0)-1.0)
                            print("SHORT: %s" % event.time)
                            self.events_queue.put(
                                    SignalEvent(self.tickers[0], "SLD", self.qty)
                                    )
                            self.invested = "SHORT"
                            self.contador = 0
                            #self.entry_price = event.close_price/float(
                            #        PriceParser.PRICE_MULTIPLIER)

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
                    elif self.invested == "SHORT":
                        self.contador += 1
#                        if current_price < self.entry_price:
#                            print("CLOSING LONG: %s" % event.time)
#                            self.events_queue.put(
#                                    SignalEvent(self.tickers[0], "BOT", self.qty)
#                                    )
#                            self.invested = "NONE"
                        #if pred > self.toplimit:
                        if self.contador == self.salida:
                            print("CLOSING SHORT: %s" % event.time)
                            self.events_queue.put(
                                    SignalEvent(self.tickers[0], "BOT", self.qty)
                                    )
                            self.invested = "NONE"                        
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
                    pred = self.modelpkl.predict(self.cur_returns.reshape((1, -1)))[0]
                    # Long only strategy
                    if not self.invested and pred == 1:
                        print("LONG: %s" % event.time)
                        self.events_queue.put(
                            SignalEvent(self.tickers[0], "BOT", self.qty)
                        )
                        self.invested = True
                    if self.invested and pred == -1:
                        print("CLOSING LONG: %s" % event.time)
                        self.events_queue.put(
                            SignalEvent(self.tickers[0], "SLD", self.qty)
                        )
                        self.invested = False

