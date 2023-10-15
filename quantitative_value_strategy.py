import yfinance as yf
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period = 'max')

#datatime index
sp500.index

#cleaning
sp500.plot.line(y="Close", use_index = True)
del sp500['Dividends']
del sp500['Stock Splits']
sp500['Tomorrow'] = sp500['Close'].shift(-1)
sp500['Target'] = (sp500['Tomorrow'] >sp500['Close']).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
print(sp500)

#train
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score
preds = model.predict(test[predictors])
preds = pd.Series(preds,index = test.index)

combined = pd.concat([test["Target"], preds], axis = 1)
combined.plot()
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined
def backtest(data, model, predictors, start = 2500, step = 250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

horizons = [2,5,60,250,1000]
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    #ratio of today's close price divided by rolling average close price over past horizon days 
    sp500[f'Close_Ratio_{horizon}'] = sp500['Close'] / rolling_averages['Close']
    #looks at past few days and counts number of times stock price went up 
    sp500[f'Trend_{horizon}'] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [f'Close_Ratio_{horizon}',f'Trend_{horizon}']

sp500 = sp500.dropna()


model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)
#modify to predict_proba : probability that the stock price will go up
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds>=.6] = 1
    preds[preds<.6] = 0
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined
predictions = backtest(sp500, model, new_predictors)