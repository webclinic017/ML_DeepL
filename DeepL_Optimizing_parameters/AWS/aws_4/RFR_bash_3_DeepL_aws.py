# intraday_ml_backtest.py

from datetime import datetime

from DeepL_fit_class_aws import Fitting_DeepL
import os

from collections import deque
import pandas as pd
import numpy as np


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
    #current_directory = os.getcwd()        
    #ts.to_csv(current_directory + "/"+ "ts" + ".csv", sep = ",")

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
     #'percent_factor' and down no more
     #then 'percent_factor*up_down_factor'
        up = percent_factor/sigma
        down = percent_factor*up_down_factor/sigma
    
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
        ts["Up"] = down_tot & up_tot
        
    # Convert True/False into 1 and 0
        ts["Up"] = ts["Up"].astype(int)
   
####################    SHORT  ##########################   
        down_cols_short = [
                ts["Lookforward%s" % str(i+1)] < -up
                for i in range(0, lookforward_minutes)
                ]
        up_cols_short = [
                ts["Lookforward%s" % str(i+1)] < down
                for i in range(0, lookforward_minutes)
                ]
        
        down_tot_short = down_cols_short[0]
        for c in down_cols_short[1:]:
            down_tot_short = down_tot_short | c
        up_tot_short = up_cols_short[0]
        for c in up_cols_short[1:]:
            up_tot_short = up_tot_short & c
        ts["Down"] = down_tot_short & up_tot_short
        ts["Down"] = ts["Down"].astype(int)
        ts["Down"].replace(to_replace=1, value=2, inplace=True)

        ts["UpDown"] = ts["Up"] | ts["Down"]
        
        current_directory = os.getcwd()        
        ts.to_csv(current_directory + "/"+ "ts" + ".csv", sep = ",")
        
        print(ts)

    
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
# los: -percent_factor < retornos < percent_factor => 0
#      retornos > percent_factor => 1
#      retornos < -percent_factor => 2
        #print(ts)
        current_directory = os.getcwd()        
        ts.to_csv(current_directory + "/"+ "ts" + ".csv", sep = ",")
            
    return ts


###############################################################################


if __name__ == "__main__":


    csv_filepath = "/Users/yoelvis/Interactive_Brokers/MASTER_Data/"
    #csv_filepath = "/Users/yoelvisorozco/Interactive_Brokers/MASTER_Data/"
    #csv_filepath = "/home/ec2-user/"

#    f = open(csv_filepath + "0labels", "r")
#    data = f.read()
#    f.close()
#    data = data.split()

    #tickers = data[ticklist]
    tickers = ["ORCL"]

    n_jobs = 1
    #lookback_minutes = 17
    lookforward_minutes = 3
    
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
                        
                        ts_file = current_directory + "/" + "ts.csv"
                        ts = pd.read_csv(ts_file, index_col="Timestamp", parse_dates=True)
##                        ts.to_csv(current_directory + "/"+ "ts_test" + ".csv", sep = ",")

#                        ts = create_up_down_dataframe(csv_filepath2, lookback_minutes, 
#                                                      lookforward_minutes, lags,
#                                                      0.5, percent_factor, skip)
                        
                        for Conv_layers in [0]:
                            for learning in [0.1]:
                                for kernel_size in [3]:
                                    for batch_size in [2048]:
                                        for conv_nodes in [32]:
                                            for LSTM_layers in [5]:
                                                for lstm_nodes in [128]:                                
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

                            
    
