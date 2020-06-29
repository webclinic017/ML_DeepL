# intraday_ml_backtest.py

from datetime import datetime

from qstrader import settings
from qstrader.compat import queue
from qstrader.event import SignalEvent, EventType
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.position_sizer.naive import NaivePositionSizer
#from qstrader.price_handler.iq_feed_intraday_csv_bar import IQFeedIntradayCsvBarPriceHandler
from qstrader.price_parser import PriceParser
from qstrader.risk_manager.example import ExampleRiskManager
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.strategy.base import AbstractStrategy
from qstrader.trading_session import TradingSession

from qstrader.execution_handler.ib_simulated import IBSimulatedExecutionHandler
from qstrader.compliance.compliance_live import LiveCompliance

from DeepL_strategy import IntradayMachineLearningPredictionStrategy
from iq_feed_intraday_csv_bar_skip import IQFeedIntradayCsvBarPriceHandler

from DeepL_fit_class import Fitting_DeepL
import os
import multiprocessing
#from subprocess import Popen

#from make_histograms import Make_hist
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import comb
from sklearn.metrics import mean_squared_error as mse
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def run(config, testing, tickers, csv_filepath, modelfile, start_date, end_date,
        lags, title, folder_name, model, return_win, percent_factor, salida, skip):
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
        end_date, skip
    )

    # Use the ML Intraday Prediction Strategy
    model_pickle_file = pklfile
    strategy = IntradayMachineLearningPredictionStrategy(
        tickers, events_queue, model_pickle_file, lags, model,
        return_win, percent_factor, salida, skip
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



def create_up_down_dataframe(csv_filepath, lookback_minutes, 
                                      lookforward_minutes, lags,
                                      up_down_factor, percent_factor, skip):
    ts = pd.read_csv(csv_filepath,
                     names=[
                             "Timestamp", "Open", "High", "Low",
                             "Close", "Volume"
                             ],
                             index_col="Timestamp", parse_dates=True
        )
# Filter on start/end dates
# YOE comentado para que cree ts con todos los valores disponibles
    #if self.start_date is not None:
    #    ts = ts[ts.index >= self.start_date]
    #if self.end_date is not None:
    #    ts = ts[ts.index <= self.end_date]

# Drop the non-essential columns
    ts.drop(
            [
            "Open", "Low", "High",
            "Volume"
            ],
            axis=1, inplace=True
            )

    print("Yoe1")
# Create the lookback and lookforward shifts
    for i in range(0, lookback_minutes):
        ts["Lookback%s" % str(i+1)] = ts["Close"].shift(skip*(i+1))
    for i in range(0, lookforward_minutes):
        ts["Lookforward%s" % str(i+1)] = ts["Close"].shift(-(skip*(i+1)))

    ts.dropna(inplace=True)
    
# Para chequear la continuidad en "Lookbacks" analizamos la continuidad de $lags pasos atras en "Lookback0"
# Si posteriormente se quiere analizar la continuidad de "Lookforwards" se analizan $lags pasos adelante en "Lookback0"
    print("Yoe2")

#    cur_time = deque((lags + 1)*[None], (lags + 1))
#    delta = np.zeros(lags)
    delete = []
#    cont1 = 0
#    cont2 = 0
    for k in range(skip):
        cur_time = deque((lags + 1)*[None], (lags + 1))
        delta = np.zeros(lags)
        cont1 = 0
        cont2 = 0
        for i in ts["Lookforward1"].index:
            if (cont1 + k) % skip == 0:
                cur_time.appendleft(i)
                cont2 += 1
                if cont2 > lags:
                    for j in range(lags):
                        a = str(cur_time[j] - cur_time[j+1])
                        #print(a)
                        try:
                            a = datetime.strptime(a, "%w days %H:%M:%S") # using %w because parse do not understand
                        except:                                          # 0, 1, 2, 3, ... but 01, 02, 03 ...
                            try:            
                                a = datetime.strptime(a, "7 days, %H:%M:%S")
                            except:
                                try:
                                    a = datetime.strptime(a, "8 days, %H:%M:%S")
                                except:
                                    try:
                                        a = datetime.strptime(a, "9 days, %H:%M:%S")
                                    except:
                                        a = datetime.strptime(a, "%d days, %H:%M:%S")                                        
                        delta[j] = int(a.strftime("%M"))
                        #delta[j] = int(a.strftime("%S"))
                    if not all(x == skip for x in delta):
                    #if not all(x == 30 for x in delta):
                        delete.append(i)
            cont1 += 1
    ts.drop(delete, axis = 0, inplace=True)
    #print(delete)

    print("Yoe3")
    # TEST
    current_directory = os.getcwd()        
    ts.to_csv(current_directory + "/"+ "ts" + ".csv", sep = ",")

    #test1 = datetime(2008, 2, 11, 15, 52, 0)
    #test2 = datetime(2008, 2, 12, 9, 36, 0)
    #print(delete)
    #ts.drop(delete, axis = 0, inplace=True)
    #for i in ts["Close"].index:
    #    if i > test1 and i < test2:
    #        print(ts.loc[i])
    #a = ts.tail(1).index
    #if a > test1:
    #    print(a)


# Adjust all of these values to be percentage returns
    
#    ts["Lookback0"] = ts["Close"].pct_change()*100.0
#    for i in range(0, lookback_minutes):
#        ts["Lookback%s" % str(i+1)] = ts["Lookback%s" % str(i+1)
#                ].pct_change()*100.0
#    for i in range(0, lookforward_minutes):
#        ts["Lookforward%s" % str(i+1)] = ts["Lookforward%s" % str(i+1)
#                ].pct_change()*100.0
#    ts.dropna(inplace=True)

# Yoe La forma Original de los retornos fue cambiada para evitar los retornos
# de un dia para otro: (las filas de ts son continuas pero las columnas no)

    ts["Lookback0"] = ((ts["Close"]/ts["Lookback1"])-1.0)*100.0

    sigma = np.std(ts["Lookback0"])
    
    print("Yoe, std: ", sigma)
    
    ts["Lookback0"] = ((ts["Close"]/ts["Lookback1"])-1.0)*100.0/sigma

    for i in range(1, lookback_minutes):
        ts["Lookback%s" % str(i)] = ((ts["Lookback%s" % str(i)]/ts["Lookback%s" % str(i+1)])-1.0)*100.0/sigma
# Yoe: Tiene que ser en reverso porque los Lookforwards se van modificando: 
    if lookforward_minutes > 1:
        for i in reversed(range(2, lookforward_minutes + 1)):
            ts["Lookforward%s" % str(i)] = ((ts["Lookforward%s" % str(i)]/ts["Lookforward%s" % str(i-1)])-1.0)*100.0/sigma
    ts["Lookforward1"] = ((ts["Lookforward1"]/ts["Close"])-1.0)*100.0/sigma
            
########################################################################################    

    if up_down_factor is not None and percent_factor is not None:
     #Determine if the stock has gone up at least by
     #'up_down_factor' x 'percent_factor' and down no more
     #then 'percent_factor'
        up = up_down_factor*percent_factor
        down = percent_factor
    
        # Create the list of True/False entries for each date
        # as to whether the up/down logic is true
        down_cols = [
                ts["Lookforward%s" % str(i+1)] > -down
                for i in range(0, lookforward_minutes)
                ]
        up_cols = [
                ts["Lookforward%s" % str(i+1)] > up
                for i in range(0, lookforward_minutes)
                ]
            
        # Carry out the bitwise and, as well as bitwise or
        # for the down and up logic
        down_tot = down_cols[0]
        for c in down_cols[1:]:
            down_tot = down_tot & c
        up_tot = up_cols[0]
        for c in up_cols[1:]:
            up_tot = up_tot | c
        ts["UpDown"] = down_tot & up_tot
        #ts["UpDown"] = np.sign(ts["Lookforward1"])
        
    # Convert True/False into 1 and 0
        ts["UpDown"] = ts["UpDown"].astype(int)
        ts["UpDown"].replace(to_replace=0, value=-1, inplace=True)
    print("Yoe4")
    
#######################################################################################
    
    if up_down_factor is None and percent_factor is not None:
#YOE
        up = percent_factor/sigma
        up_cols = ts["Lookforward1"] > up  # devuelve un array de True or False
        down_cols = ts["Lookforward1"] < -up
        ts["Up"] = up_cols
        ts["Up"] = ts["Up"].astype(int)
        ts["Down"] = down_cols
        ts["Down"] = ts["Down"].astype(int)
#        ts["Down"].replace(to_replace=1, value=-1, inplace=True)
        ts["Down"].replace(to_replace=1, value=2, inplace=True)
        ts["UpDown"] = ts["Up"] | ts["Down"]
# los: -percent_factor < retornos > percent_factor => 0
#      retornos > percent_factor => 1
#      retornos < -percent_factor => 2
        #print(ts)
            
    return ts


###############################################################################


if __name__ == "__main__":
    # Configuration data
    testing = False
    config = settings.from_file(
        settings.DEFAULT_CONFIG_FILENAME, testing
    )

######### FITTING parameters #########

#    csv_filepath = "/Users/yoelvis/Interactive_Brokers/MASTER_Data/"
    csv_filepath = "/Users/yoelvisorozco/Interactive_Brokers/MASTER_Data/"

#    f = open(csv_filepath + "0labels", "r")
#    data = f.read()
#    f.close()
#    data = data.split()

    #tickers = data[ticklist]
    tickers = ["AMAT_test"]

    n_jobs = 1
    #lookback_minutes = 17
    lookforward_minutes = 2
    
    skip = 1
    #up_down_factor = 2
    percent_factor = 0.07        
    model = "ConvLSTM_Clasifier"

    iterate = True

    current_directory = os.getcwd()
    for i in tickers:
        folder_name = i + "_" + model + "_Tune_Params"
        putmodel = current_directory + "/" + folder_name + "/"
        try:
            os.mkdir(current_directory + "/" + folder_name + "/")
        except:
            print("Folder Excists")
        csv_filepath2 = csv_filepath + i + ".csv"        
        if iterate:
            if model == "ConvLSTM_Clasifier":
                for lags in [10]:
                    lookback_minutes = lags
                    for skip in [1]:
                        print("Importing and creating CSV DataFrame...")
                        ts = create_up_down_dataframe(csv_filepath2, lookback_minutes, 
                                                      lookforward_minutes, lags,
                                                      None, percent_factor, skip)
                        
                        for Conv_layers in [4]:
                            for learning in [0.1]:
                                for kernel_size in [3]:
                                    for batch_size in [128]:
                                        for conv_nodes in [64]:
                                            for lstm_nodes in [64]:
                                                for LSTM_layers in [3]:                                
                                                    name = model + "_lags%s_skip%s_nConv%s_nLSTM%s_learning%s_batch_size%s_conv_nodes%s_lstm_nodes%s_kernel_size%s" %(
                                                            lags, skip, Conv_layers, LSTM_layers, learning, batch_size, conv_nodes, lstm_nodes, kernel_size)   # esta es la carpeta que contendra los pkl and png
                                                    start_date = None
                                                    end_date = datetime(2017, 3, 1) #datetime(2017, 8, 16)
    #                                                n_jobs = 4
                                                    
                                                    fit = Fitting_DeepL(ts, putmodel, 
                                                                     lags, 
                                                                     start_date, end_date, model,
                                                                     i, percent_factor, skip,
                                                                     Conv_layers, LSTM_layers, learning, 
                                                                     batch_size, conv_nodes, lstm_nodes,
                                                                     kernel_size)
                                                    try:
                                                        os.remove(putmodel + i + ".pkl")
                                                    except:
                                                        print("No pkl file to delete")
                                                    fit.fitting()
                fit.history.to_csv(putmodel + "history.csv")
#                        fit.histogram()
#                        for salida in [1]:
#                            tick = [i]
#                            start_date = datetime(2016, 1, 1)
#                            end_date = None
#                            return_win = fit.X_train
#                            modelfile = putmodel + i + ".pkl"
#                            
#                            title = [name + "_percent_factor%s" %percent_factor + "_salida%s" %salida + "_" + i]                           
#                            run(config, testing, tick, csv_filepath, modelfile, start_date, end_date, 
#                                lags, title, folder_name, model, return_win, percent_factor, salida, skip)

                            
      
                                        
                                        
                                    


                                
        else:

            #max_depth = 5
            # min_samples_split = 20
            lags = 6
            skip = 5
            model = "RFR"
            lookback_minutes = lags
            percent_factor = 0.067
                        
            print("Importing and creating CSV DataFrame...") 
            ts = create_up_down_dataframe(csv_filepath2, lookback_minutes, 
                                          lookforward_minutes, lags, putmodel, 
                                          None, percent_factor, skip)

            n_jobs = 1
#            name = model + "_lags%s_min_samples_split%s_max_depth%s" %(
#                    lags, min_samples_split, max_depth)   # esta es la carpeta que contendra los pkl and png
            name = model + "_lags%s_poly%s_min_samples_leaf%s_skip%s" %(
                    lags, skip)   # esta es la carpeta que contendra los pkl and png

            start_date = None
            end_date = datetime(2016, 1, 1)
            
            fit = Fitting_ML(ts, putmodel,
                             n_jobs, lags, 
                             start_date, end_date, model,
                             i, None, skip)
                             #min_samples_split, max_depth, i, None, skip)
            try:
                os.remove(putmodel + i + ".pkl")
            except:
                print("No pkl file to delete")
            fit.fitting()
            fit.histogram()
            print("***********************************************")
            print("Used parameters:")
            print("***********************************************")
                                
        ############# OUT of sample test ################
    
            pklfile = putmodel + i + ".pkl"
            tick = [i]
                                                    
            start_date = datetime(2016, 1, 1)
            end_date = None
                            
    #       X_train = fit.X_train[-fit.window:]
    #       X_train_array = [[X_train.iloc[j][0], X_train.iloc[j][1]] for j in range(0, fit.window)]        
    #       return_win = deque(maxlen = fit.window)
    #
    #       for k in range(0, fit.window):
    #       return_win.append(X_train_array[k])
    
            n_jobs = 1    
            return_win = fit.X_train

            paral = 0
            for salida in [1]: #, 2]:
                for percent_factor in [0.067]: # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    paral += 1
                    if paral <= 100: 
                        title = [name + "_percent_factor%s" %percent_factor + "_salida%s" %salida + "_" + i]                           
                        run(config, testing, tick, csv_filepath, pklfile, start_date, end_date, 
                            lags, title, folder_name, model, return_win, percent_factor, salida, skip)
                        os.rename(putmodel + i + ".png", putmodel + i + "_percent_factor%s" %percent_factor + "_salida%s" 
                                  %salida + "_skip%s" %skip + ".png")


            
#            start_date = None
#            end_date = None
#            n_jobs = 1
#            fit = Fitting_ML(csv_filepath, putpkl, random_state, n_estimators, 
#                             n_jobs, lags, 
#                             start_date, end_date, model, 
#                             min_samples_split, max_depth, i)
#            try:
#                os.remove(putpkl + i + ".pkl")
#            except:
#                print("No pkl file to delete")
#            fit.fitting()

