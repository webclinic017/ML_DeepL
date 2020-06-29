# intraday_ml_strategy.py

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy

from itertools import cycle

from datetime import datetime, timedelta, time


class IntradayMachineLearningPredictionStrategy(AbstractStrategy):
  """
  Requires:
  tickers - The list of ticker symbols
  events_queue - A handle to the system events queue
  """
    def __init__(
        self, tickers, events_queue, model_pickle_file, strat_set,
        open_position):
        
        self.tickers = tickers
        self.events_queue = events_queue
        self.lags = strat_set["lags"]
        self.RF = strat_set["model"]
        mean_trained = strat_set["mean"]
        sigma_trained = strat_set["sigma"]
        percent_factor = strat_set["percent_factor"]
        
        self.salida = strat_set["salida"]
        self.contador = 0
        
        if open_position[0] == tickers[0]:
            if open_position[1] < 0:
                self.invested = "SHORT"
            elif open_position[1] > 0:
                self.invested = "LONG"
            self.qty = abs(open_position[1])
        else:
            self.invested = "NONE"
            self.prev_invested = None
            self.qty = None

        self.cur_prices = np.zeros(self.lags+1)
        self.cur_returns = np.zeros(self.lags)
        self.minutes = 0
        self.modelpkl = joblib.load(model_pickle_file)
        
        self.percent_factor = percent_factor
#        self.return_win = return_win
        self.entry_price = 0
        self.conta = cycle(range(0,500))
        #self.botlimit = None
        #self.toplimit = None
#        pred = self.modelpkl.predict(return_win)
#        mean = np.mean(pred)
#        sigma = np.std(pred)
        self.botlimit = mean_trained - self.percent_factor*sigma_trained
        self.toplimit = mean_trained + self.percent_factor*sigma_trained


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
        inittime = datetime(2000, 1, 1, 9, 30).time()
        finaltime = (datetime(2000, 1, 1, 15, 59, 00) - timedelta(minutes=self.salida)).time()
        if datetime.now().time() >= inittime and datetime.now().time() <= finaltime:
            if self.RF == "RFR":        
  """
  Calculate the intraday machine learning 
  prediction strategy.
  """
                if event.type == EventType.BAR:
                    self._update_current_returns(event)
                    self.minutes += 1
                    gap = event.ask - event.bid
                    # Allow enough time to pass to populate the 
                    # returns feature vector
                    if self.minutes > (self.lags + 2) and event.close_price > 0: #>0 por seguridad
                        pred = self.modelpkl.predict(self.cur_returns.reshape((1, -1)))[0]
                        # Long only strategy
                        if self.invested == "NONE" and gap <= 200000:
                            if pred > self.toplimit:
                                self.qty = int(round(30000/(event.close_price/float(
                                        PriceParser.PRICE_MULTIPLIER)),0)-1.0)
                                print("LONG: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "buy", self.qty)
                                        )
                                self.prev_invested = "NONE"
                                self.invested = "LONG"
                                self.contador = 0
                                #self.entry_price = event.close_price/float(
                                #        PriceParser.PRICE_MULTIPLIER)
                            elif pred < self.botlimit:
                                self.qty = int(round(30000/(event.close_price/float(
                                        PriceParser.PRICE_MULTIPLIER)),0)-1.0)
                                print("SHORT: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "sell", self.qty)
                                        )
                                self.prev_invested = "NONE"
                                self.invested = "SHORT"
                                self.contador = 0
                                #self.entry_price = event.close_price/float(
                                #        PriceParser.PRICE_MULTIPLIER)
    
                        elif self.invested == "LONG":
                            self.contador += 1
                            if self.contador >= self.salida:
                                print("CLOSING LONG: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "sell", self.qty)
                                        )
                                self.prev_invested = "LONG"
                                self.invested = "NONE"
                        elif self.invested == "SHORT":
                            self.contador += 1
                            if self.contador >= self.salida:
                                print("CLOSING SHORT: %s" % event.time)
                                self.events_queue.put(
                                        SignalEvent(self.tickers[0], "buy", self.qty)
                                        )
                                self.prev_invested = "SHORT"
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



        #Testing section #########################
#        self.minutes += 1
#        if self.invested == "NONE":
#            if self.minutes > 1:
#                self.qty = int(round(30000/(event.close_price/float(
#                        PriceParser.PRICE_MULTIPLIER)),0)-1.0)
#                print("LONG: %s" % event.time)
#                self.events_queue.put(
#                        SignalEvent(self.tickers[0], "buy", self.qty)
#                        )
#                self.prev_invested = "NONE"
#                self.invested = "LONG"
#                self.contador = 0
#                #self.entry_price = event.close_price/float(
#                #        PriceParser.PRICE_MULTIPLIER)
#
#        elif self.invested == "LONG":
#            self.contador += 1
#            if self.contador == 4:
#                print("CLOSING LONG: %s" % event.time)
#                self.events_queue.put(
#                        SignalEvent(self.tickers[0], "sell", self.qty)
#                        )
#                self.prev_invested = "LONG"
#                self.invested = "NONE"
