from datetime import datetime
import numpy as np

from qstrader import settings
from qstrader.strategy.base import AbstractStrategy
from qstrader.event import SignalEvent, EventType
from qstrader.compat import queue
from qstrader.trading_session import TradingSession
from qstrader.price_handler import opening_intraday_csv_bar


class Opening_strategy(AbstractStrategy):
    """
    A testing strategy that simply purchases (longs) an asset
    upon first receipt of the relevant bar event and
    then holds until the completion of a backtest.
    """
    def __init__(
        self, ticker, events_queue, model):

#        base_quantity=100
        self.ticker = ticker
        self.events_queue = events_queue
#        self.bars = 0
        self.invested = False
        self.close_premarket = []
        self.volume_premarket = []
        self.model = model
        self.tipo = None        
        self.base_quantity = None

    def calculate_signals(self, event):
        if (
            event.type in [EventType.BAR, EventType.TICK] and
            event.ticker == self.ticker
        ):
            if not self.invested: # and self.bars == 0:
                if event.time.time() <= datetime(1900, 1, 1, 9, 30).time():
                    self.close_premarket.append(event.close_price)
                    self.volume_premarket.append((event.high_price + event.low_price)/2 * event.volume)
                if event.time.time() == datetime(1900, 1, 1, 9, 30).time():
                    abre = event.open_price
                    self.base_quantity = int(90000/abre)
                    standard = np.std(self.close_premarket)
                    std = standard*100/abre
                    media = np.mean(self.close_premarket)
                    entry_point = (abre - media)*100/abre
                    vol_price = np.sum(self.volume_premarket)
                    X = [[entry_point, std, vol_price]]
                    predict = self.model.predict_proba(X)
                    if predict[0][1] > 0.9:
                        signal = SignalEvent(self.ticker, "BOT", 
                                             suggested_quantity=self.base_quantity)
                        self.tipo = "long"
                        self.events_queue.put(signal)
                        self.invested = True
                    if predict[0][0] > 0.9:
                        signal = SignalEvent(self.ticker, "SLD", 
                                             suggested_quantity=self.base_quantity)
                        self.tipo = "short"
                        self.events_queue.put(signal)
                        self.invested = True
#                    self.bars += 1
            if self.invested: # and self.bars == 0:
                if event.time.time() == datetime(1900, 1, 1, 10, 30).time():
                    if self.tipo == "long":
                        signal = SignalEvent(self.ticker, "SLD", 
                                             suggested_quantity=self.base_quantity)                       
                        self.tipo = None
                        self.events_queue.put(signal)
                        self.invested = False

                    if self.tipo == "short":
                        signal = SignalEvent(self.ticker, "BOT", 
                                             suggested_quantity=self.base_quantity)
                        self.tipo = None
                        self.events_queue.put(signal)
                        self.invested = False

                        
                    


