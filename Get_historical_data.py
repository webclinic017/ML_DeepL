# intraday_ml_backtest.py

#import datetime


from ibapi.common import *  # para los errores
from ibapi.contract import *  # para los detalles de contratos
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sys import exit
import os
from qstrader.price_handler.Conexion_yoe import conexion
from ibapi.common import *  # para los errores
from ibapi.contract import *  # para los detalles de contratos
import pandas as pd


def run(cliente):

#
# Defining the Conection to TWS ######################################################
#    
    current_directory = os.getcwd()
#    app = conexion(tickers=["ANYTHING"], port = 7497, cliente = cliente, 
#                   currentDir = current_directory) # devuelto en: currentTime    
#    queryTime = (datetime.today() - 
#                 timedelta(days=180)).strftime("%Y%m%d %H:%M:%S")
    a = datetime.today()
    start = datetime(a.year, a.month, a.day - 1, 0, 0, 0)  # al comienzo del dia 12 AM 
    for j in ["JBLU", "MU", "FOXA", "LNG", "EVHC", "MSCC", "X", "C", "FL", "GM", "SKX", "CF", "SEE", "TOL", "HFC", "HOG", "TER"]:
        app = conexion(tickers=["ANYTHING"], port = 7497, cliente = cliente, 
                       currentDir = current_directory) # devuelto en: currentTime    
        Id = 5000
        app.Historical = current_directory + "/" + j        
        contract_1 = Contract()
        contract_1.symbol = j
        contract_1.secType = "STK"
        contract_1.currency = "USD"
        contract_1.exchange = "SMART"
        contract_1.primaryExchange = "ISLAND"                
        for i in reversed(range(20)):        
            newId = Id + i
            app.HistDataEnd = None
            tiempo = (start - relativedelta(months = 6*i)).strftime("%Y%m%d %H:%M:%S")
            app.reqHistoricalData(newId, contract_1, tiempo, 
                                  "7 M", "1 min", "TRADES", 1, 1, False, [])
            conti = False
            while not conti:
                time.sleep(1)
                chk = app.HistDataEnd
                if chk is not None or app.errorCode == 162 or app.errorCode == 200: #Historical Market Data Service error message:HMDS query returned no data:
                    conti = True
            time.sleep(3)
            app.cancelHistoricalData(newId)
            time.sleep(3)
#            resto = i % 5
#            if resto == 0:
#                pandasdata = pd.DataFrame(app.rows_list, columns=['Tiempo', 'Open', 'High', 'Low', 'Close', 'Volume'])
#                pandasdata.set_index('Tiempo', inplace = True)
#                pandasdata.drop_duplicates(inplace = True)
#                pandasdata.sort_index(inplace = True)
#                pandasdata.to_csv(current_directory + "/" + j + "_" + str(i) + ".csv", sep = ',', header = False, date_format='%Y%m%d%H%M%S')

        pandasdata = pd.DataFrame(app.rows_list, columns=['Tiempo', 'Open', 'High', 'Low', 'Close', 'Volume'])
        pandasdata.set_index('Tiempo', inplace = True)
        pandasdata.drop_duplicates(inplace = True)
        pandasdata.sort_index(inplace = True)
        pandasdata.to_csv(current_directory + "/" + j + ".csv", sep = ',', header = False, date_format='%Y%m%d%H%M%S')
        #end_date = datetime(2018, 2, 15, 0, 0, 0)
        #test = pandasdata[pandasdata.index > end_date]
        #print(test)
        app.rows_list = []
        app.disconnect()

if __name__ == "__main__":
    run(20)
    
