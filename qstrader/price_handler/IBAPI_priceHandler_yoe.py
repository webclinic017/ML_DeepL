import os

import pandas as pd

from qstrader.price_parser import PriceParser
from .base import AbstractBarPriceHandler
from qstrader.event import BarEvent

from ibapi.common import *  # para los errores
from ibapi.contract import *  # para los detalles de contratos

from datetime import datetime, date, timedelta
import time
from qstrader.price_handler.Conexion_yoe import conexion
from itertools import cycle




class IBAPI_yoe(AbstractBarPriceHandler):
#class Prueba():
    """
    IBAPI is designed to fetch data from IB
    for each requested financial instrument and stream those to
    the provided events queue as BarEvents.
    """
    def __init__(
        self, events_queue,
        init_tickers, app, reqId,
        start_date=None, end_date=None,
        calc_adj_returns=False
    ):
        self.events_queue = events_queue
#        self.continue_backtest = True  # Yoe, la condicion ahora es "end_session_time"
        self.app = app
        self.tickers = {}
        self.tickers_data = {}
        self.init_tickers = init_tickers
        print("Ciclo", reqId)
        self.tickers_loop_id = cycle(reqId)
        self.tickers_loop = cycle(init_tickers)
        self.calc_adj_returns = calc_adj_returns
        if self.calc_adj_returns:
            self.adj_close_returns = []

        if init_tickers is not None:
            for ticker in init_tickers:
                self.subscribe_ticker(ticker)

    def subscribe_ticker(self, ticker):
        """
        Subscribes the price handler to a new ticker symbol.
        """
        if ticker not in self.tickers:
            ticker_id = next(self.tickers_loop_id)
            ticker = next(self.tickers_loop)
            data = self.get_data_from_IB(ticker_id)
            close = PriceParser.parse(data[1])
            adj_close = PriceParser.parse(data[2])
            timestamp = data[0]
                
            ticker_prices = {
                "close": close,
                "adj_close": adj_close,
                "timestamp": timestamp
            }
            self.tickers[ticker] = ticker_prices
        else:
            print(
                "Could not subscribe ticker %s "
                "as is already subscribed." % ticker
            )
    def _create_event(self, index, period, ticker, row):
        """
        Obtain all elements of the bar from a row of dataframe
        and return a BarEvent
        """
        open_price = PriceParser.parse(row["Open"])
        high_price = PriceParser.parse(row["High"])
        low_price = PriceParser.parse(row["Low"])
        close_price = PriceParser.parse(row["Close"])
        adj_close_price = PriceParser.parse(row["Adj Close"])
        volume = int(row["Volume"])
        bid = PriceParser.parse(row["bid"])
        ask = PriceParser.parse(row["ask"])
        bev = BarEvent(
            ticker, index, period, open_price,
            high_price, low_price, close_price,
            volume, adj_close_price, bid, ask
        )
        return bev

    def _store_event(self, event):
        """
        Store price event for closing price and adjusted closing price
        """
        ticker = event.ticker
        # If the calc_adj_returns flag is True, then calculate
        # and store the full list of adjusted closing price
        # percentage returns in a list
        # TODO: Make this faster
        if self.calc_adj_returns:
            prev_adj_close = self.tickers[ticker][
                "adj_close"
            ] / float(PriceParser.PRICE_MULTIPLIER)
            cur_adj_close = event.adj_close_price / float(
                PriceParser.PRICE_MULTIPLIER
            )
            self.tickers[ticker][
                "adj_close_ret"
            ] = cur_adj_close / prev_adj_close - 1.0
            self.adj_close_returns.append(self.tickers[ticker]["adj_close_ret"])
        self.tickers[ticker]["close"] = event.close_price
        self.tickers[ticker]["adj_close"] = event.adj_close_price
        self.tickers[ticker]["timestamp"] = event.time

    def stream_next(self):
        """
        Place the next BarEvent onto the event queue.        
        """
        ticker_id = next(self.tickers_loop_id)
        ticker = next(self.tickers_loop)
        period = 60  # Seconds in a minute

        data = self.get_data_from_IB(ticker_id)

        index = data[0]
        row = {
                'Open': data[3],
                'High': data[3],
                'Low': data[3],
                'Close': data[3],
                'Adj Close': data[3],
                'Volume': 100,
                'bid': data[1],
                'ask': data[2]}        
        
        bev = self._create_event(index, period, ticker, row)
        
        # Store event
        self._store_event(bev)
        # Yoe
        # Aqui se actualiza el array de array tickers
        
        self.events_queue.put(bev)
#        time.sleep(2)  # probando
        
    def get_data_from_IB(self, ticker_id):        
        bid = None
        ask = None
        close = None
        adj_close = None

        bid = self.app.precios[ticker_id][0]
        ask = self.app.precios[ticker_id][1]
        close = self.app.precios[ticker_id][2]
        adj_close = self.app.precios[ticker_id][2]

        if bid is None or ask is None or close is None or adj_close is None:
            bid = -1.0
            ask = -1.0
            close = -1.0
            adj_close = -1.0


#        while bid is None or ask is None or close is None or adj_close is None or \
#            close == -1 or close == 0.0 or adj_close == -1 or adj_close == 0.0:
#
#            bid = self.app.precios[ticker_id][0]
#            ask = self.app.precios[ticker_id][1]
#            close = self.app.precios[ticker_id][2]
#            adj_close = self.app.precios[ticker_id][2]

           
#        conti = False
#        while not conti:
#            #print("Aquiiiiiii")
#            bid = self.app.precios[ticker_id][0]
#            ask = self.app.precios[ticker_id][1]
#            close = self.app.precios[ticker_id][2]
#            adj_close = self.app.precios[ticker_id][2]
#          #  print("No esta emitiendo precios .... (This is in PriceHandler)")
#            if bid is None or ask is None or close is None or adj_close is None:
#                conti = False
#            else:
#                conti = True
#        if close == -1 or close == 0.0 or adj_close == -1 or adj_close == 0.0:
#                            
#            conti = False
#            tiempo = datetime.now()
#            while not conti:
#                if datetime.now() > tiempo:
#                    print("#########################################")
#                    print("Yoe, close price or adj_close is 0.0 or -1.")
#                    print("     I will continue iterating every 1s")
#                    print("     untill receive a proper price")
#                    print("     Local time: %s" %datetime.now())
#                    print("#########################################")
#                    tiempo = tiempo + timedelta(seconds=30)
#                time.sleep(1)
#                bid = self.app.precios[ticker_id][0]
#                ask = self.app.precios[ticker_id][1]
#                close = self.app.precios[ticker_id][2]
#                adj_close = self.app.precios[ticker_id][2]
#                if close == -1 or close == 0.0 or adj_close == -1 or adj_close == 0.0:
#                    conti = False
#                else:
#                    conti = True




        hora = str(datetime.now())
        hora = datetime.strptime(hora[11:19], '%H:%M:%S')
        hora = datetime.time(hora)

        fecha = date.today()
        # se necesita en este formato:
        timestamp = datetime.combine(fecha, hora) 
        
        precio = [timestamp, bid, ask, adj_close]
        
        time.ctime()
        a = time.strftime('%b %d, %Y, %l:%M%p')
        print("Yoeee", a, self.init_tickers[0], precio[1:4], "Yoeee")
        return precio

#if __name__ == '__main__':
#    Prueba(events_queue=queue.Queue())
        
