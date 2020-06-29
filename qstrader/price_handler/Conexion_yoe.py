from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from threading import Thread
from ibapi.common import *  # para los errores
from ibapi.contract import *  # para los detalles de contratos

from sys import exit
from datetime import datetime, timedelta
import time
from ibapi.utils import iswrapper



class conexion(EWrapper, EClient):  
    def __init__(self, tickers, port, cliente, currentDir):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper = self)

        ipaddress = "127.0.0.1"
        portid = port   # Gateway: 4002, TWS: 7497
        clientid = cliente

###################################################################

        self.precios = []
        self.hora = []
        self.nextValidOrderId = None
        
        self.Id_File = currentDir + "/" + "NuevoId_file"

        for i in range(5000):
            self.precios.append([None, None, None])
            self.hora.append(None)

        self.commission = None
        self.fill_price = None
        self.remaining = None
        self.errorCode = None
        self.marketDataType = None
        self.open_position = [None, None]
        self.tickers = tickers

###################################################################       

        self.connect(ipaddress, portid, clientid)
        if self.errorCode == 502:  # el socket no puede ser abierto
            print("########################################")
            print("  Yoe, This Socket cannot be opened or ")
            print("       IB Wateway is closed")
            print("########################################")
            exit()
        thread = Thread(target = self.run)
        thread.start()

        setattr(self, "_thread", thread)

###################################################################
        #self.reqIds(-1)
        tiempo = datetime.now()    # esperando confirmacion de la conexion con IB Gateway
        conti = None
        while conti is None:
            if datetime.now() > tiempo + timedelta(seconds=5):
                print("####################################################################")
                print("Yoe, 5 seconds passed waiting to stablish connection with IB wateway")
                print("     Let's wait 5s more ...")
                print("####################################################################")
                tiempo = tiempo + timedelta(seconds=5)
            conti = self.nextValidOrderId
#        print("AQUI connect", self.nextValidOrderId)

        print("########################################")
        print("  Yoe, connection with IB Gateway is ok,")
        print("       nextValidOrderId: %s" %self.nextValidOrderId)
        print("########################################")
              
              
    def nuevoId(self):
        try:
            f = open(self.Id_File, "r")
            nuevo = int(f.read())
            f.close()
            f = open(self.Id_File, "w+")
            f.write(str(nuevo+1))
            f.close()
        except IOError:
            # If not exists, create the file
            f = open(self.Id_File, 'w+')
            f.write("1")
            f.close()
            nuevo = 1
        return nuevo      

###############################################################################
###############################################################################        

#   Triggered by reqMktData in Main code
    def marketDataType(self, reqId, marketDataType): # para el tipo de dato solicitado: reqMarketDataType()
        super().marketDataType(reqId, marketDataType)
        #print("MarketDataType. ", reqId, "Type:", marketDataType)
        self.marketDataType = marketDataType

#   Triggered by reqMktDataType in Main code
    def tickPrice(self, reqId, tickType, price,
                  attrib):
        super().tickPrice(reqId, tickType, price, attrib)        
#        print("Tick Price. Ticker Id:", reqId, "tickType:", tickType, "Price:", 
#              price, "CanAutoExecute:", attrib.canAutoExecute, 
#              "PastLimit", attrib.pastLimit)
        # YOE: reqId es el Idnumber de la orden ejecutada en reqMktData
        # tickType = 0: Bid Size. Number of contracts or lots offered at the bid price
        # tickType = 1: Bid Price. Highest priced bid for the contract.
        # tickType = 2: Ask Price. Lowest price offer on the contract.
        # tickType = 3: Ask Size. Number of contracts or lots offered at the ask price.
        # tickType = 4: Last Price. Last price at which the contract traded.
        if (tickType == 1):
            self.precios[reqId][0] = price
        if (tickType == 2):
            self.precios[reqId][1] = price
        if (tickType == 4):
            self.precios[reqId][2] = price
        #    print('reqId: ', reqId, 'tickType: ', tickType, 'price: ', price)

#   Triggered by placeOrder in Execution_handler
    def openOrder(self, orderId, contract, order, 
                orderState):
        super().openOrder(orderId, contract, order, orderState)
#        print("OpenOrder. ID:", orderId, contract.symbol, contract.secType,
#                "@", contract.exchange, ":", order.action, order.orderType,
#                order.totalQuantity, orderState.commission)
        self.commission = orderState.commission
#        print('Yoe', orderId, contract.symbol, order.action, 'Comision', self.commission)

#   Triggered by placeOrder in Execution_handler
    def orderStatus(self, orderId, status, filled,
                        remaining, avgFillPrice, permId,
                        parentId, lastFillPrice, clientId,
                        whyHeld, mktCapPrice):
        super().orderStatus(orderId, status, filled, remaining,
                            avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
#        print("OrderStatus. Id:", orderId, "Status:", status, "Filled:", filled,
#                "Remaining:", remaining, "AvgFillPrice:", avgFillPrice,
#                "PermId:", permId, "ParentId:", parentId, "LastFillPrice:",
#                lastFillPrice, "ClientId:", clientId, "WhyHeld:", whyHeld)
        self.fill_price = avgFillPrice
        self.remaining = remaining
#        print('Yoe', 'Fill Price:', self.fill_price)

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
#        logging.debug("setting nextValidOrderId: %d", orderId)
        self.nextValidOrderId = orderId

#   Triggered by errorCode in Execution_handler
    def error(self, reqId, errorCode, errorString):
        super().error(reqId, errorCode, errorString)
        #print("Error. Id: ", reqId, " Code: ", errorCode, " Msg: ", errorString)
        self.errorCode = errorCode

        
##################################### 
# Requesting Account Update
#####################################
        
#    def updateAccountValue(self, key, val, currency,
#                           accountName):
#        super().updateAccountValue(key, val, currency, accountName)
#        print("UpdateAccountValue. Key:", key, "Value:", val, 
#              "Currency:", currency, "AccountName:", accountName)
#
        
#   Triggered by reqAccountUpdates in Main code
    def updatePortfolio(self, contract, position, 
                        marketPrice, marketValue, 
                        averageCost, unrealizedPNL, 
                        realizedPNL, accountName):
        super().updatePortfolio(contract, position, marketPrice, marketValue, 
             averageCost, unrealizedPNL, realizedPNL, accountName)
#        print("UpdatePortfolio.", contract.symbol, "", contract.secType, "@", 
#              contract.exchange, "Position:", position, "MarketPrice:", marketPrice, 
#              "MarketValue:", marketValue, "AverageCost:", averageCost, 
#              "UnrealizedPNL:", unrealizedPNL, "RealizedPNL:", realizedPNL, 
#              "AccountName:", accountName)
        if contract.symbol in self.tickers and position != 0.0:
            self.open_position[0] = contract.symbol
            self.open_position[1] = position
#
#    def updateAccountTime(self, timeStamp):
#        super().updateAccountTime(timeStamp)
#        print("UpdateAccountTime. Time:", timeStamp)
#
#    def accountDownloadEnd(self, accountName):
#        super().accountDownloadEnd(accountName)
#        print("Account download finished:", accountName)
 
##############################


   



