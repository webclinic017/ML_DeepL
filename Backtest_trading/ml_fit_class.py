  #
  # intraday_ml_model_fit.py
  # Traning the model using Random Forest
  #
import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor)
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Fitting_ML():
    def __init__(self,
                 csv_filepath,
                 putpkl,
                 random_state, 
                 n_estimators,
                 n_jobs, 
                 lookback_minutes, 
                 lookforward_minutes, 
                 lags,
                 start_date, 
                 end_date, 
                 #up_down_factor,
                 #percent_factor,
                 model,
                 min_samples_split,
                 max_depth,
                 ticker
                 ):
        
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        self.lookback_minutes = lookback_minutes
        self.lookforward_minutes = lookforward_minutes
                
        self.up_down_factor = 2
        self.percent_factor = 0.1
        self.model = model
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        self.histfile = putpkl + ticker
        self.pklfile = putpkl + ticker + ".pkl"
        self.metrics = putpkl + "score_" + ticker
        
        csv_filepath = csv_filepath + ticker + ".csv"
        print("Importing and creating CSV DataFrame...") 
        ts = self.create_up_down_dataframe(csv_filepath)

        print("Creating train/test split of data...")

        self.X_train = ts[ts.index < end_date][["Lookback%s" % str(i) for i in range(1, lags+1)]]
        self.X_test = ts[ts.index >= end_date][["Lookback%s" % str(i) for i in range(1, lags+1)]]
  #
  # Random Forest Regresor
  #
        if self.model == "RFR":
            self.y_train = ts[ts.index < end_date]["Lookback0"]
            self.y_test = ts[ts.index >= end_date]["Lookback0"]
  #
  # Random Forest Clasifier
  #
        elif self.model == "RFC":                                      
            self.y_train = ts[ts.index < end_date]["UpDown"]
            self.y_test = ts[ts.index >= end_date]["UpDown"]
        
        self.ticker = ticker
        self.lags = lags
        self.window = 500
        
        self.R2 = None

    def fitting(self):
                
        #csv_filepath = self.csv_filepath + ticker + ".csv"

        #print("Importing and creating CSV DataFrame...")    
        #ts = self.create_up_down_dataframe(csv_filepath)
                        
        # Use the training-testing split with 70% of data in the
        # training data with the remaining 30% of data in the testing
            #X_train, X_test, y_train, y_test = train_test_split(
            #        X, y, test_size=0.25, random_state=self.random_state, shuffle=False
            #        )
        print("Fitting classifier model...")

  #
  # Evaluating the model
  #

        if self.model == 'mla':
            model = LinearDiscriminantAnalysis()
        elif self.model == 'BG':
            model = BaggingClassifier(
                    base_estimator=DecisionTreeClassifier(),
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                    )
        elif self.model == 'GBC':
            model = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state
                    )
        elif self.model == 'RFC':
            model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    min_samples_split=self.min_samples_split
                    #max_depth = self.max_depth
                    )
            model.fit(self.X_train, self.y_train)
            print("Outputting metrics...")
            print("Hit-Rate: %s" % model.score(self.X_test, self.y_test))
            print("%s\n" % confusion_matrix(model.predict(self.X_test), self.y_test))
                
        elif self.model == 'RFR':
            model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth
                    )
            model.fit(self.X_train, self.y_train)
            print("Outputting metrics...")
            self.R2 = model.score(self.X_test, self.y_test)
            text1 = "R^2, defined as (1 - u/v): %s\n" % self.R2
            print(text1)
            y_pred = model.predict(self.X_test)
            text2 = "Minimun Square Error: %s\n" % mean_squared_error(self.y_test, y_pred)
            print(text2)
            
            mean = np.mean(y_pred)
            sigma = np.std(y_pred)
            text3 = "predicted mean, sigma: %s, %s\n" %(mean, sigma)
            mean = np.mean(self.y_train)
            sigma = np.std(self.y_train)
            text4 = "trained mean, sigma: %s, %s\n" %(mean, sigma)
            
            f = open(self.metrics, 'w+')
            f.write("Outputting metrics...")
            f.write(text1)
            f.write(text2)
            f.write(text3)
            f.write(text4)
            f.close()

        print("Pickling model...")
        
        
        joblib.dump(model, self.pklfile)

  #
  # Histograms of the training data and predicted data to compare the 
  # statistical prediction power 
  #
        
    def histogram(self):
        
        mean_train = np.mean(self.y_train)
        sigma_train = np.std(self.y_train)
        
        
        modelpkl = joblib.load(self.pklfile)
        #y_pred = modelpkl.predict(self.X_train[-self.window:])
        y_pred = modelpkl.predict(self.X_train)

    
    #####################################################################
    
        mean = np.mean(y_pred)
        sigma = np.std(y_pred)
        print("predicted train set: mean, sigma: ", mean, sigma)
        x = np.linspace(-5*sigma_train, 5*sigma_train, 100)
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(2,1,1)
        ax.hist(y_pred, bins=30, range = (-5*sigma, 5*sigma), normed=True, 
                label="Predicted train set:\n mean = %.4f\n std = %.4f" %(mean, sigma))
        ax.plot(x, mlab.normpdf(x, mean, sigma), 'r-')
        
        ax.legend(loc='best')
        ax.set_title('%s_lags%s_min_samples_split%s_max_depth%s' %(
                self.ticker, self.lags, self.min_samples_split, self.max_depth), y=1.08)
        ax.set_xlim(-4*sigma_train, 4*sigma_train)
        
        ax1 = ax.twiny()
        ax1.set_xlim(-4*sigma_train, 4*sigma_train)
        ax1.set_xticks(np.arange(-4*sigma, 4.1*sigma, sigma))
        ax1.set_xticklabels([r"${} \sigma$".format(i) for i in range(-4,5)])
                        
#        fig.savefig(self.histfile + "_predicted.png", dpi=150, bbox_inches='tight')
        
    ###################################################################
    
        #y = self.y_train[-self.window:]

        print("train set: mean, sigma: ", mean_train, sigma_train)

        x = np.linspace(-5*sigma_train, 5*sigma_train, 100)
        
        #fig = plt.figure()
        bx = fig.add_subplot(2,1,2)
        bx.hist(self.y_train, bins=30, range = (-5*sigma_train, 5*sigma_train), normed=True, 
                label="Train set:\n mean = %.4f\n std = %.4f" %(mean_train, sigma_train))
        bx.plot(x, mlab.normpdf(x, mean_train, sigma_train), 'r-')
        
        bx.legend(loc='best')
        bx.set_xlim(-4*sigma_train, 4*sigma_train)
        
        bx1 = bx.twiny()
        bx1.set_xlim(-4*sigma_train, 4*sigma_train)
        bx1.set_xticks(np.arange(-4*sigma_train, 4.1*sigma_train, sigma_train))
        bx1.set_xticklabels([r"${} \sigma$".format(i) for i in range(-4,5)])
        
        plt.tight_layout()

        fig.savefig(self.histfile + "_train_set.png", dpi=150, bbox_inches='tight')

  ####################################################################
  ####################################################################
  #   OUT OF THE SAMPLING (predicted values)
    
        mean_test = np.mean(self.y_test)
        sigma_test = np.std(self.y_test)
        
        
        modelpkl = joblib.load(self.pklfile)
        #y_pred = modelpkl.predict(self.X_train[-self.window:])
        y_pred = modelpkl.predict(self.X_test)

    
  #####################################################################
    
        mean = np.mean(y_pred)
        sigma = np.std(y_pred)
        print("predicted test set: mean, sigma: ", mean, sigma)
        x = np.linspace(-5*sigma_test, 5*sigma_test, 100)
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(2,1,1)
        ax.hist(y_pred, bins=30, range = (-5*sigma, 5*sigma), normed=True, 
                label="Predicted test set:\n mean = %.4f\n std = %.4f" %(mean, sigma))
        ax.plot(x, mlab.normpdf(x, mean, sigma), 'r-')
        
        ax.legend(loc='best')
        ax.set_title('%s_lags%s_min_samples_split%s_max_depth%s' %(
                self.ticker, self.lags, self.min_samples_split, self.max_depth), y=1.08)
        ax.set_xlim(-4*sigma_test, 4*sigma_test)
        
        ax1 = ax.twiny()
        ax1.set_xlim(-4*sigma_test, 4*sigma_test)
        ax1.set_xticks(np.arange(-4*sigma, 4.1*sigma, sigma))
        ax1.set_xticklabels([r"${} \sigma$".format(i) for i in range(-4,5)])
                        
#        fig.savefig(self.histfile + "_predicted.png", dpi=150, bbox_inches='tight')
        
  ###################################################################
    
        #y = self.y_train[-self.window:]

        print("test set: mean, sigma: ", mean_test, sigma_test)

        x = np.linspace(-5*sigma_test, 5*sigma_test, 100)
        
        #fig = plt.figure()
        bx = fig.add_subplot(2,1,2)
        bx.hist(self.y_train, bins=30, range = (-5*sigma_test, 5*sigma_test), normed=True, 
                label="Test set:\n mean = %.4f\n std = %.4f" %(mean_train, sigma_test))
        bx.plot(x, mlab.normpdf(x, mean_train, sigma_test), 'r-')
        
        bx.legend(loc='best')
        bx.set_xlim(-4*sigma_test, 4*sigma_test)
        
        bx1 = bx.twiny()
        bx1.set_xlim(-4*sigma_test, 4*sigma_test)
        bx1.set_xticks(np.arange(-4*sigma_test, 4.1*sigma_test, sigma_test))
        bx1.set_xticklabels([r"${} \sigma$".format(i) for i in range(-4,5)])
        
        plt.tight_layout()

        fig.savefig(self.histfile + "_test_set.png", dpi=150, bbox_inches='tight')



        #plt.show()

    def create_up_down_dataframe(self, csv_filepath):
#     csv_filepath,
#     lookback_minutes=30,
#     lookforward_minutes=5,
#     up_down_factor=2.0,
#     percent_factor=0.01,
#     start=None, end=None
#):
  """
  Creates a Pandas DataFrame that imports and calculates
  the percentage returns of an intraday OLHC ticker from disk.
    
  'lookback_minutes' of prior returns are stored to create
  a feature vector, while 'lookforward_minutes' are used to
  ascertain how far in the future to predict across.
    
  The actual prediction is to determine whether a ticker
  moves up by at least 'up_down_factor' x 'percent_factor',
  while not dropping below 'percent_factor' in the same period.
    
  i.e. Does the stock move up 1% in a minute and not down by 0.5%?
    
  The DataFrame will consist of 'lookback_minutes' columns for feature
  vectors and one column for whether the stock adheres to the "up/down"
  rule, which is 1 if True or 0 if False for each minute.
  """
        ts = pd.read_csv(csv_filepath,
                         names=[
                                 "Timestamp", "Open", "High", "Low",
                                 "Close", "Volume"
                                 ],
                                 index_col="Timestamp", parse_dates=True
            )
  # Filter on start/end dates
  # Commented to create a ts with all available values
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
    
  # Create the lookback and lookforward shifts
        for i in range(0, self.lookback_minutes):
            ts["Lookback%s" % str(i+1)] = ts["Close"].shift(i+1)
        for i in range(0, self.lookforward_minutes):
            ts["Lookforward%s" % str(i+1)] = ts["Close"].shift(-(i+1))
        ts.dropna(inplace=True)

  # Adjust all of these values to be percentage returns
        ts["Lookback0"] = ts["Close"].pct_change()*100.0
        for i in range(0, self.lookback_minutes):
            ts["Lookback%s" % str(i+1)] = ts["Lookback%s" % str(i+1)
                    ].pct_change()*100.0
        for i in range(0, self.lookforward_minutes):
            ts["Lookforward%s" % str(i+1)] = ts["Lookforward%s" % str(i+1)
                    ].pct_change()*100.0
        ts.dropna(inplace=True)
        #print(ts)

  # Determine if the stock has gone up at least by
  # 'up_down_factor' x 'percent_factor' and down no more
  # then 'percent_factor'
        up = self.up_down_factor*self.percent_factor
        down = self.percent_factor

  # Create the list of True/False entries for each date
  # as to whether the up/down logic is true
        down_cols = [
                ts["Lookforward%s" % str(i+1)] > -down
                for i in range(0, self.lookforward_minutes)
                ]
        up_cols = [
                ts["Lookforward%s" % str(i+1)] > up
                for i in range(0, self.lookforward_minutes)
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
        return ts
