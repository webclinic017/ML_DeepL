from qstrader.execution_handler.base import AbstractExecutionHandler
from qstrader.event import (FillEvent, EventType)
from qstrader.price_parser import PriceParser

from ibapi.order import *

import time
from datetime import datetime, timedelta
from itertools import cycle

import threading


class IB_execution_yoe(AbstractExecutionHandler):
    
    def __init__(self, events_queue, price_handler, compliance=None, app=None,
                 contract_dict=None, currentDir = None, Id=None, strategy=None):
        """
        Initialises the handler, setting the event queue
        as well as access to local pricing.

        Parameters:
        events_queue - The Queue of Event objects.
        """
        self.events_queue = events_queue
        self.price_handler = price_handler
        self.compliance = compliance
        
        self.app = app
        self.contract_dict = contract_dict
        self.trades_file = currentDir + "/" + "trades_file"
        
        self.tickers_loop_id = cycle(Id)
        
        self.strategy = strategy
        

#    def calculate_ib_commission(self, quantity, fill_price):
#        """
#        Calculate the Interactive Brokers commission for
#        a transaction. This is based on the US Fixed pricing,
#        the details of which can be found here:
#        https://www.interactivebrokers.co.uk/en/index.php?f=1590&p=stocks1
#        """
#        commission = min(
#            0.5 * fill_price * quantity,
#            max(1.0, 0.005 * quantity)
#        )
#        return PriceParser.parse(commission)

    def execute_order(self, event):
        """
        Converts OrderEvents into FillEvents "naively",
        i.e. without any latency, slippage or fill ratio problems.

        Parameters:
        event - An Event object with order information.
        """
        ticker_id = next(self.tickers_loop_id)
        if event.type == EventType.ORDER:
            # Obtain values from the OrderEvent
            timestamp = self.price_handler.get_last_timestamp(event.ticker)
            ticker = event.ticker
            action = event.action
            quantity = event.quantity

#            order = Order()
#            order.action = action
#            order.orderType = "MKT"
#            order.totalQuantity = quantity

# this is not working: (Unsupported order type for this exchange and security type.)           
#            order = Order()
#            order.action = action
#            order.orderType = "PEG MID"
#            order.totalQuantity = quantity
#            #order.auxPrice = 0.05 #offset, it seems to be higher than 0.05
#            #order.lmtPrice = limitPrice
            
            bid = self.app.precios[ticker_id][0]
            ask = self.app.precios[ticker_id][1]
            last_trd = self.app.precios[ticker_id][2]
            limitPrice = abs((ask + bid)/2.0)
            
            if action == "buy":
                limitPrice = min(last_trd, ask)
            elif action == "sell":
                limitPrice = max(last_trd, bid)
#            limitPrice = last_trd
            
            order = Order()
            order.action = action
            order.orderType = "LMT"
            order.totalQuantity = quantity
            order.lmtPrice = limitPrice
            order.sweepToFill = True
            
            print('################# submitiendo la orden ##################')
            
            self.app.commission = None
            self.app.fill_price = None
            commission = None
            fill_price = None
            orderid = self.app.nuevoId()
            print("Orden ID", orderid)
            self.app.placeOrder(orderid, self.contract_dict[ticker], 
                            order)           
#            commission = self.calculate_ib_commission(quantity, fill_price)
            conti = False
            conti2 = False
            canceled = False
            while not conti:
                commission = self.app.commission
                fill_price = self.app.fill_price
                if commission == None or fill_price == None or commission == 1.7976931348623157e+308 or fill_price == 0.0:
                    conti = False
                    if self.app.errorCode == 202:
                        print("******")
                        print("Yoe, Order Canceled")
                        print("******")
                        canceled = True
                        conti = True
                        #exit()
                    elif self.app.errorCode == 399:
                        tiempo = datetime.now()
                        while not conti2:
                            commission = self.app.commission
                            fill_price = self.app.fill_price
                            if commission == None or fill_price == None or commission == 1.7976931348623157e+308 or fill_price == 0.0:
                                conti2 = False
                                time.sleep(1)  # para que no sea tan seguido
                                if datetime.now() > tiempo:
                                    tiempo = tiempo + timedelta(minutes=60)
                                    print("#########################################")
                                    print("Yoe, It seems to be that the RTH has passed.")
                                    print("     I will continue iterating every 1s")
                                    print("     untill the oder be submitted.")
                                    print("     Local time: %s" %datetime.now())
                                    print("#########################################")
                            else:
                                conti2 = True
                                conti = True
                else:
                    conti = True
                    
            commission = PriceParser.parse(commission)
            fill_price = PriceParser.parse(fill_price)
            print('real_comision: ', commission, self.app.commission, 
                  'precio: ', fill_price, self.app.fill_price)

            if order.action == 'buy':
                order.action = 'BOT'
            if order.action == 'sell':
                order.action = 'SLD'

#             Obtain the fill price
#            if self.price_handler.istick():
#                bid, ask = self.price_handler.get_best_bid_ask(ticker)
#                if event.action == "BOT":
#                    fill_price = ask
#                else:
#                    fill_price = bid
#            else:
#                close_price = self.price_handler.get_last_close(ticker)
#                fill_price = close_price

            # Set a dummy exchange and calculate trade commission
            exchange = "ARCA"

            # Create the FillEvent and place on the events queue
            if not canceled: 
                try:
                    f = open(self.trades_file, "r")
                    #nuevo = int(f.read())
                    f.close()
                    f = open(self.trades_file, "a")
                    texto = "%s %s %s %s %s %s %s\n" %(timestamp, ticker, action, quantity, exchange, 
                                                       fill_price/float(PriceParser.PRICE_MULTIPLIER),
                                                       commission/float(PriceParser.PRICE_MULTIPLIER)) 
                    f.write(texto)
                    f.close()
                except IOError:
                    # If not exists, create the file
                    f = open(self.trades_file, 'w+')
                    texto = "%s %s %s %s %s %s %s\n" %(timestamp, ticker, action, quantity, exchange, 
                                                       fill_price/float(PriceParser.PRICE_MULTIPLIER),
                                                       commission/float(PriceParser.PRICE_MULTIPLIER)) 
                    f.write(texto)
                    f.close()    
            if not canceled: 
                fill_event = FillEvent(
                        timestamp, ticker,
                        action, quantity, 
                        exchange, fill_price, 
                        commission)
                
                self.events_queue.put(fill_event)

                if self.compliance is not None:
                    self.compliance.record_trade(fill_event)
            if canceled:
                self.strategy.invested = None
                self.strategy.contador = 0
