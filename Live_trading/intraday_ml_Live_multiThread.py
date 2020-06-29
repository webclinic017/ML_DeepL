# intraday_live trading.py

from qstrader import settings
from qstrader.compat import queue
from qstrader.event import SignalEvent, EventType
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.position_sizer.naive import NaivePositionSizer
#  IBAPI_yoe swill be used
from qstrader.price_parser import PriceParser
from qstrader.risk_manager.example import ExampleRiskManager
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.strategy.base import AbstractStrategy
from qstrader.trading_session import TradingSession

from intraday_ml_strategy import IntradayMachineLearningPredictionStrategy

from qstrader.price_handler.Conexion_yoe import conexion
from qstrader.price_handler.IBAPI_priceHandler_yoe import IBAPI_yoe
from qstrader.execution_handler.IB_execution_handler_yoe import IB_execution_yoe
from qstrader.compliance.compliance_live import LiveCompliance

from ibapi.common import *  # For errors
from ibapi.contract import *  # For contract details
import time
from datetime import datetime, timedelta
from sys import exit
import os

import threading
import time


def run(config, testing, tickers, cliente, strat_set):

    events_queue = queue.Queue()
    initial_equity = 30000.0

    end_session_time = datetime(2018, 2, 23, 16, 1, 00)
    if end_session_time < datetime.now():
        print("########################################")
        print("  Yoe, Increase \"end_section time\" please")
        print("########################################")
        exit()
        
    start_date = None
    end_date = None

  #
  # Defining up to 3 contracts in case Mean Reversing Strategies were used in the future
  #
    
    contract_1 = Contract()
    contract_1.symbol = tickers[0]
    contract_1.secType = "STK"
    contract_1.currency = "USD"
    contract_1.exchange = "SMART"
    contract_1.primaryExchange = "ISLAND"        

    contract_2 = Contract()
    if len(tickers) > 1:
        contract_2.symbol = tickers[1]
    contract_2.secType = "STK"
    contract_2.currency = "USD"
    contract_2.exchange = "SMART"
    contract_2.primaryExchange = "ISLAND"

    contract_3 = Contract()
    if len(tickers) > 2:
        contract_3.symbol = tickers[2]
    contract_3.secType = "STK"
    contract_3.currency = "USD"
    contract_3.exchange = "SMART"
    contract_3.primaryExchange = "ISLAND"

    contract = [contract_1, contract_2, contract_3]

  #
  # Defining the Conection to TWS ######################################################
  #    
    current_directory = os.getcwd()
    app = conexion(tickers=tickers, port = 7497, cliente = cliente, 
                   currentDir = current_directory) # devuelto en: currentTime
  #    
  #   Checking open positions on the market ############################################
  #    
    app.reqAccountUpdates(True, "DU931045")

    tiempo = datetime.now()
    continua = False
    while not continua:
        open_position = app.open_position
        if open_position[0] != None or datetime.now() > tiempo + timedelta(seconds=3):
            continua = True
        time.sleep(0.5)
    if open_position[0] == tickers[0]:
        print(" *******************************************************")
        print("")
        print(" There is an open possition for %s, please check in the " %tickers[0])
        print(" market whether it is convenient to proced automatically")
        print("")
        print(" *******************************************************")

    app.reqAccountUpdates(False, "DU931045")

  #
  # Requesting real time data from the Market ##########################################
  #
    reqId = [None for i in range(len(tickers))]
    for i in range(len(tickers)):
        reqId[i] = app.nuevoId()
        app.reqMarketDataType(1)
        app.reqMktData(reqId[i], contract[i], genericTickList='', snapshot=False,
                           regulatorySnapshot=False, mktDataOptions=[])
        print("request Id: ", reqId[i])
    
  #
  # Defining modules ###################################################################
  # 

    price_handler = IBAPI_yoe(events_queue, 
                              init_tickers=tickers, app=app, reqId=reqId)

#    start_date = datetime.datetime(2015, 1, 1)
#    end_date = datetime.datetime(2016, 9, 30)
#    price_handler = IQFeedIntradayCsvBarPriceHandler(
#        csv_dir, events_queue, tickers, start_date=start_date
#    )
    contract_dict = {
            contract_1.symbol: contract_1,
            contract_2.symbol: contract_2,
            contract_3.symbol: contract_3
            }

    LiveCompl = LiveCompliance(config, tickers)

  # Use the ML Intraday Prediction Strategy
    
    model_pickle_file = current_directory + "/" + tickers[0] + "_Full_time.pkl"
    strategy = IntradayMachineLearningPredictionStrategy(
        tickers, events_queue, model_pickle_file, strat_set, open_position
    )

    execution_handler=IB_execution_yoe(events_queue, 
                                               price_handler, 
                                               compliance=LiveCompl,
                                               app=app, 
                                               contract_dict=contract_dict,
                                               currentDir = current_directory,
                                               Id=reqId,
                                               strategy = strategy)

  # Use the Naive Position Sizer where
  # suggested quantities are followed
    position_sizer = NaivePositionSizer()

  # Use an example Risk Manager
    risk_manager = ExampleRiskManager()

  # Use the default Portfolio Handler
    portfolio_handler = PortfolioHandler( 
        PriceParser.parse(initial_equity),
        events_queue, price_handler,
        position_sizer, risk_manager
    )

    title = [
        tickers[0] + "Machine Learning_Long_lags3"
    ]
  # Use the Tearsheet Statistics
    statistics = TearsheetStatistics(
        config, portfolio_handler,
        title=title,
        periods=int(252*6.5*60)  # Minutely periods
    )

  # Set up the backtest
    
    backtest = TradingSession(
        config, strategy, tickers,
        initial_equity, start_date, end_date, events_queue, 
        session_type="live", end_session_time=end_session_time,
        price_handler=price_handler, portfolio_handler=portfolio_handler,
        compliance=None, position_sizer=position_sizer,
        execution_handler=execution_handler, risk_manager=None,
        statistics=statistics, sentiment_handler=None,
        title=None, benchmark=None
    )
    results = backtest.start_trading(testing=testing)

    return results
    for i in range(len(tickers)):
        app.cancelMktData(i)
        app.disconnect()
    app.f.close()

class myThread (threading.Thread):
   def __init__(self, config, testing, tickers, cliente, strat_set):
      threading.Thread.__init__(self)
      self.name = tickers[0]
      self.config = config
      self.tickers = tickers
      self.testing = testing
      self.cliente = cliente
      self.strat_set = strat_set
   def run(self):
      print("Starting " + self.name)
      run(self.config, self.testing, self.tickers, self.cliente, self.strat_set)
      print("Exiting " + self.name)


if __name__ == "__main__":
    # Configuration data
    testing = False
    config = settings.from_file(
        settings.DEFAULT_CONFIG_FILENAME, testing
    )

    strat_set = []
    #AABA: Max_depth = 5, Min_Sample_Split = 20, lags = 2
    strat_set.append({"name": "AABA",
                         "lags": 2, 
                         "model": "RFR", 
                         "mean": 0.000277739952749, 
                         "sigma": 0.0627807987649,
                         "percent_factor": 0.4,
                         "salida": 1})

    #AMAT: Max_depth = 5, Min_Sample_Split = 20, lags = 2
    strat_set.append({"name": "AMAT",
                         "lags": 2, 
                         "model": "RFR", 
                         "mean": 0.000263872269205, 
                         "sigma": 0.0608197538993,
                         "percent_factor": 0.2,
                         "salida": 1})

    #GILD: Max_depth = 10, Min_Sample_Split = 20, lags = 3
    strat_set.append({"name": "GILD",
                         "lags": 3, 
                         "model": "RFR", 
                         "mean": 0.00017495461192, 
                         "sigma": 0.0629220321243,
                         "percent_factor": 0.2,
                         "salida": 1})

    #JBLU: Max_depth = 5, Min_Sample_Split = 20, lags = 3
    strat_set.append({"name": "JBLU",
                         "lags": 3, 
                         "model": "RFR", 
                         "mean": 0.0003732853245, 
                         "sigma": 0.0676913151563,
                         "percent_factor": 0.6,
                         "salida": 1})

    #MU: Max_depth = 12, Min_Sample_Split = 500, lags = 2
    strat_set.append({"name": "MU", 
                         "lags": 2, 
                         "model": "RFR", 
                         "mean": 0.00047080119655, 
                         "sigma": 0.0834142717394,
                         "percent_factor": 0.9,
                         "salida": 1})

    #FOXA: Max_depth = 12, Min_Sample_Split = 20, lags = 3
    strat_set.append({"name": "FOXA", 
                         "lags": 3, 
                         "model": "RFR", 
                         "mean": 0.000234524243621, 
                         "sigma": 0.0872021765117,
                         "percent_factor": 0.2,
                         "salida": 1})

    #AMAG: Max_depth = 16, Min_Sample_Split = 500, lags = 2
    strat_set.append({"name": "AMAG", 
                         "lags": 2, 
                         "model": "RFR", 
                         "mean": 0.000244185380118, 
                         "sigma": 0.0645733356563,
                         "percent_factor": 0.8,
                         "salida": 1})

    #LNG: Max_depth = 5, Min_Sample_Split = 10, lags = 3
    strat_set.append({"name": "LNG", 
                         "lags": 3, 
                         "model": "RFR", 
                         "mean": 0.000331003114527, 
                         "sigma": 0.031719602925,
                         "percent_factor": 0.3,
                         "salida": 1})

    
    hilos = [None for i in range(len(strat_set))]
    ticker_list = []
    for i in strat_set:
        ticker_list.append(i["name"])
    cont = 0
    for i in ticker_list:
        # Create new threads
        tickers = [i]
        cliente = cont + 1
        hilos[cont] = myThread(config, testing, tickers, cliente, strat_set[cont])
        # Start new Threads
        hilos[cont].start()
        cont = cont + 1
        time.sleep(0.5)
    print("Exiting Main Thread")



#    filename = None
#    tickers = [ticker_list[0]]
#    run(config, testing, tickers, 1)
