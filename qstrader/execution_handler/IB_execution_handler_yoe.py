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

# cuando sea con dinero real se debe poner entre el bid y el ask
# porque de esa forma estariamos mejorando el bid y el ask en el libro
# y es muy probable que se ejecute:            
            #limitPrice = abs((ask + bid)/2.0)

            if action == "buy":
                limitPrice = min(last_trd, ask)
            elif action == "sell":
                limitPrice = max(last_trd, bid)
                            
            order = Order()
            order.action = action
            order.orderType = "LMT"
            order.totalQuantity = quantity
            order.lmtPrice = limitPrice
            order.sweepToFill = True
            
            print('############## submitiendo orden ###############')
            print('%s %s, limitPrice: %s, bid: %s, ask: %s, close: %s' %(action, ticker, limitPrice, bid, ask, last_trd))
            
            self.app.commission = None
            self.app.fill_price = None
            self.app.remaining = None
            self.app.errorCode = None
            commission = None
            fill_price = None
            remaining = None
            error = None
            orderid = self.app.nuevoId()
            #print("Orden ID", orderid)
            self.app.placeOrder(orderid, self.contract_dict[ticker], 
                            order)
            
            complete = None
            continua = False
            cancel = 0
            timeout = time.time() + 30
            while not continua:
                time.sleep(1)
                remaining = self.app.remaining
                error = self.app.errorCode
                if error == 202:  # Order Canceled: An active order on the IB server was cancelled
                    # Ningun parcial filled por error
                    if remaining is None or remaining == quantity:
                        cancel = 1
                        continua = True
                        print("Orden Cancelada por el Broker ")
                        print("*********************************************************")
                    #Parcial filled, el resto error
                    else:
                        complete = False
                        continua = True
                        print("Orden submatida parcialmente, resto cancelada por el broker ")
                        
                elif remaining == 0.0:
                    complete = True
                    continua = True
                    print("Orden submatida Completamente")

                elif time.time() > timeout:
                    if remaining is None or remaining == quantity:
                        cancel = 2
                        continua = True
                        print("Orden Cancelada despues de 30 segundos")
                        print("*********************************************************")
                    else:
                        complete = False
                        continua = True
                        print("Orden submatida parcialmente, resto cancelada despues de 30 segundos ")
                    self.app.cancelOrder(orderid)
            
            # Set a dummy exchange and calculate trade commission
            exchange = "ARCA"  #nedded for statistics

            prev = self.strategy.prev_invested
            if cancel == 0:
                time.sleep(1)
                commission = self.app.commission
                fill_price = self.app.fill_price
                commission = PriceParser.parse(commission)
                fill_price = PriceParser.parse(fill_price)
                
                if action == "buy" and prev == "NONE" and not complete:
                    self.strategy.qty = quantity - remaining
                    quantity = quantity - remaining
                if action == "buy" and prev == "SHORT" and not complete:
                    self.strategy.invested = "SHORT"
                    self.strategy.qty = remaining
                    quantity = quantity - remaining
                if action == "sell" and prev == "NONE" and not complete:
                    self.strategy.qty = quantity - remaining
                    quantity = quantity - remaining
                if action == "sell" and prev == "LONG" and not complete:
                    self.strategy.invested = "LONG"
                    self.strategy.qty = remaining
                    quantity = quantity - remaining
                    
                try:
                    # If exists, appends
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
                    
            # Create the FillEvent and place on the events queue
                    
                if order.action == 'buy':
                    order.action = 'BOT'
                    action = 'BOT'
                if order.action == 'sell':
                    order.action = 'SLD'
                    action = 'SLD'
                print("OKKK", timestamp, ticker, action, quantity, exchange, 
                      commission/float(PriceParser.PRICE_MULTIPLIER), 
                      fill_price/float(PriceParser.PRICE_MULTIPLIER))
                print("*********************************************************")

                fill_event = FillEvent(
                        timestamp, ticker,
                        action, quantity, 
                        exchange, fill_price, 
                        commission)
                self.events_queue.put(fill_event)

                if self.compliance is not None:
                    self.compliance.record_trade(fill_event)

            elif cancel == 2:
                if prev == "NONE":
                    self.strategy.invested = "NONE"
                elif action == "buy" and prev == "SHORT":
                    self.strategy.invested = "SHORT"
                if action == "sell" and prev == "LONG":
                    self.strategy.invested = "LONG"
            elif cancel == 1:
                print("*****************************************")
                print(" Yoe, take some action because this order")
                print(" being canceled by the broker")
                print("*****************************************")
                
                
