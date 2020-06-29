# intraday_ml_backtest.py

import datetime

from qstrader import settings
from qstrader.compat import queue
from qstrader.event import SignalEvent, EventType
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.price_handler.iq_feed_intraday_csv_bar import IQFeedIntradayCsvBarPriceHandler
from qstrader.price_parser import PriceParser
from qstrader.risk_manager.example import ExampleRiskManager
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.strategy.base import AbstractStrategy
from qstrader.trading_session import TradingSession

from qstrader.execution_handler.ib_simulated import IBSimulatedExecutionHandler
from qstrader.compliance.compliance_live import LiveCompliance

from ml_strategy import IntradayMachineLearningPredictionStrategy

from ml_fit_class import Fitting_ML
import os
import multiprocessing
#from subprocess import Popen

#from make_histograms import Make_hist
from collections import deque

def run(config, testing, tickers, csv_filepath, pklfile, start_date, end_date,
        lags, title, folder_name, model, return_win, percent_factor, salida):
    # Set up variables needed for backtest
#    title = [
#        "Intraday AREX Machine Learning Prediction Strategy"
#    ]
    events_queue = queue.Queue()
    csv_dir = csv_filepath
    initial_equity = 30000.0

    # Use DTN IQFeed Intraday Bar Price Handler
#    start_date = datetime.datetime(2016, 1, 1)
#    end_date = datetime.datetime(2014, 3, 11)

#    start_date = datetime.datetime(2013, 1, 1)
#    end_date = datetime.datetime(2014, 3, 11)
    price_handler = IQFeedIntradayCsvBarPriceHandler(
        csv_dir, events_queue, tickers, start_date,
        end_date
    )

    # Use the ML Intraday Prediction Strategy
    model_pickle_file = pklfile
    strategy = IntradayMachineLearningPredictionStrategy(
        tickers, events_queue, model_pickle_file, lags, model,
        return_win, percent_factor, salida
    )

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

    # Use the Tearsheet Statistics
    statistics = TearsheetStatistics(
        config, portfolio_handler,
        title=title,
        periods=int(252*6.5*60)  # Minutely periods
    )

    # Set up the backtest

    compliance = LiveCompliance(config, tickers)
 
    execution_handler = IBSimulatedExecutionHandler(
	events_queue = events_queue,
	price_handler = price_handler,
	compliance = compliance)

    backtest = TradingSession(
        config, strategy, tickers,
        initial_equity, start_date, end_date,
        events_queue, title=title,
        portfolio_handler=portfolio_handler,
	compliance = compliance,
        position_sizer=position_sizer,
	execution_handler=execution_handler,
        price_handler=price_handler,
        statistics=statistics
    )
    results = backtest.start_trading(testing=testing, 
                             folder_name=folder_name)
    return results


if __name__ == "__main__":
    # Configuration data
    testing = False
    config = settings.from_file(
        settings.DEFAULT_CONFIG_FILENAME, testing
    )

######### FITTING parameters #########

    csv_filepath = "/users/PCS0202/bgs0361/Yoe_new/Master/Liquids/"

    f = open(csv_filepath + "0labels", "r")
    data = f.read()
    f.close()
    data = data.split()

    #tickers = data[ticklist]
    tickers = ["AABA"]

    random_state = 42
    n_estimators = 500
    n_jobs = 16
    lookback_minutes = 10
    lookforward_minutes = 2
    #lags = 2
    #up_down_factor = 1.5
    #percent_factor = 0.001        
    #model = "RFC"
    model = "RFR"
    #min_samples_split = 100
    #max_depth = 10

    iterate = False

    current_directory = os.getcwd()
    R2 = []
    sett = []
    for i in tickers:
        folder_name = i + "_" + model + "_lags_MinSamplesSplit_MaxDepth"
        putpkl = current_directory + "/" + folder_name + "/"
        try:
            os.mkdir(current_directory + "/" + folder_name + "/")
        except:
            print("Folder Excists")
        if iterate:
            for max_depth in [5, 10, 12, 14, 16, 18, 20, 25, 30, 35]:
                for min_samples_split in [2, 10, 20, 50, 100, 500, 1000, 3000, 5000, 10000]:
                    for lags in [2, 3, 4, 5, 6, 7, 8]:
                        sett.append((max_depth, min_samples_split, lags))                        
                        name = model + "_lags%s_min_samples_split%s_max_depth%s" %(
                                lags, min_samples_split, max_depth)   # esta es la carpeta que contendra los pkl and png
                        start_date = None
                        end_date = datetime.datetime(2016, 1, 1)
    
                        fit = Fitting_ML(csv_filepath, putpkl, random_state, n_estimators, 
                                         n_jobs, lookback_minutes, lookforward_minutes, lags, 
                                         start_date, end_date, model, 
                                         min_samples_split, max_depth, i)
                        try:
                            os.remove(putpkl + i + ".pkl")
                        except:
                            print("No pkl file to delete")
                        fit.fitting()
                        R2.append(fit.R2)
            
            maxR2 = R2.index(max(R2))
        
            max_depth = sett[maxR2][0]
            min_samples_split = sett[maxR2][1]
            lags = sett[maxR2][2]
            
            try:
                os.remove(putpkl + i + ".pkl")
            except:
                print("No pkl file to delete")

            print("***********************************************")
            print("Final Result:")
            print("R^2, defined as (1 - u/v): %s" % fit.R2)
            print("Best RFR parameters:")
            print("max_depth: %s, min_samples_split: %s, lags: %s" %(max_depth, min_samples_split, lags))
            print("***********************************************")

            
        else:
            max_depth = 5
            min_samples_split = 20
            lags = 2

            n_jobs = 1
            name = model + "_lags%s_min_samples_split%s_max_depth%s" %(
                    lags, min_samples_split, max_depth)   # esta es la carpeta que contendra los pkl and png
            start_date = None
            end_date = datetime.datetime(2016, 1, 1)
            fit = Fitting_ML(csv_filepath, putpkl, random_state, n_estimators, 
                             n_jobs, lookback_minutes, lookforward_minutes, lags, 
                             start_date, end_date, model, 
                             min_samples_split, max_depth, i)
            try:
                os.remove(putpkl + i + ".pkl")
            except:
                print("No pkl file to delete")
            fit.fitting()
            fit.histogram()
            print("***********************************************")
            print("Used parameters:")
            print("max_depth: %s, min_samples_split: %s, lags: %s" %(max_depth, min_samples_split, lags))
            print("***********************************************")
                                
        ############# OUT of sample test ################
    
            pklfile = putpkl + i + ".pkl"
            tick = [i]
                                                    
            start_date = datetime.datetime(2016, 1, 1)
            end_date = None
                            
    #       X_train = fit.X_train[-fit.window:]
    #       X_train_array = [[X_train.iloc[j][0], X_train.iloc[j][1]] for j in range(0, fit.window)]        
    #       return_win = deque(maxlen = fit.window)
    #
    #       for k in range(0, fit.window):
    #       return_win.append(X_train_array[k])
            return_win = fit.X_train

            paral = 0
            for percent_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for salida in [1, 2, 3, 4, 5]:
                    paral += 1
                    if paral <= 25: 
                        title = [name + "_percent_factor%s" %percent_factor + "_salida%s" %salida + "_" + i]                           
                        run(config, testing, tick, csv_filepath, pklfile, start_date, end_date, 
                            lags, title, folder_name, model, return_win, percent_factor, salida)
                        os.rename(putpkl + i + ".png", putpkl + i + "_percent_factor%s" %percent_factor + "_salida%s" %salida + ".png")
            
            start_date = None
            end_date = None
            n_jobs = 1
            fit = Fitting_ML(csv_filepath, putpkl, random_state, n_estimators, 
                             n_jobs, lookback_minutes, lookforward_minutes, lags, 
                             start_date, end_date, model, 
                             min_samples_split, max_depth, i)
            try:
                os.remove(putpkl + i + ".pkl")
            except:
                print("No pkl file to delete")
            fit.fitting()
        
        
#    try:
#        os.remove(pklfile)
#    except:
#        print("No pkl file to delete")

