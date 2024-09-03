Strategy Evaluation

# 1 INTRODUCTION

This project focuses on utilizing technical indicators, time series data to develop a machine learning algorithm and to determine if it out-performs a manual trading strategy that is driven based off technical indicators. For Manual Strategy we utilized three indicators: Bollinger Bands, Exponential Moving Average (EMA), and Momentum to determine when to BUY, SELL, HOLD a stock called “JPM” with in-sample and out-of-sample data. Random Forest with a Random Tree Learner was trained on in-sample stock data with our indicators, and tested on out-of-sample stock data to determine if our strategies can outperform manual strategy and our benchmark with different stocks symbols or time periods. Our Hypothesis for this project is that a Manual Strategy and Strategy Learner should outperform a Benchmark Strategy, Strategy Learner should outperform Manual Strategy Learner and that as impact increases it will decrease the number of trades and decrease portfolio returns.

# 2 INDICATOR OVERVIEW

## 2.1 BOLLINGER BAND

Bollinger bands utilize a formula to calculate the moving average of the stock prices, and the rolling standard deviation to determine the upper and lower bands which are +/- of two standard deviations of the rolling mean. The Bollinger approach is to determine the prices and if they are high or low relative to the mean share price.

Instead of having a lower and upper band, a ratio number was calculated to allow us to have one vector to determine a buy or sell as shown by the formula below:

bb_value\[t\] = (price\[t\] - SMA\[t\])/(2 \* stdev\[t\])

The reason this is a valuable indicator is because it allows us to understand the price of the stock relative to the Bollinger band. If we determine if the price is progressing to the upper Bollinger band, it is an indicator to buy, in conjunction with other relevant indicators to verify if it’s a BUY Signal. In contrast the opposite trend would be valuable to determine if it’s worth a SELL Signal.

The window size we’ve selected for both Manual and Strategy Learner is a 7 Day window. Anything larger may create wide variances in trades that are not optimal due to the ratio being underfit.

## 2.2 EMA

EMA is a derivative of simple moving average (SMA), which is intended to understand trends with the share price over specific time frames. EMA emphasizes on more recent time frames, while SMA puts an equal weight on all observed share prices. For this implementation we leveraged a calculation as follows:

![Figure0](https://github.com/dandia89/personal-projects/blob/master/DecisionTree/Figure0_EMAFormula.png)

**_Figure 1-_** EMA Formula (Chen, 2024)

We vectorized the EMA by taking the share price dividing by the EMA value and subtracted 1. The larger value indicates the share price is higher than the EMA, aka the current stock price is trading higher than the average and vice-versa for a lower value.

The window size we’ve selected for both Manual is 15 and Strategy Learner is 6 optimized based off the returns I’ve observed.

## 2.3 MOMENTUM

Momentum is an indicator that can identify the trend in stock prices by utilizing the stock price today (t) and a stock price (N) days ago. Where (N) is a stock price at a specific date in the past. The N value can change to determine how the momentum shifts with different time periods, indicating support to determine the direction of the trend of stock price.

momentum\[t\] = (price\[t\]/price\[t-N\]) – 1

The momentum indicator will have a positive value to indicate an uptrend or a buy signal, or a negative number indicating a sell or short signal. If complemented with other indicators it could be valuable to help identify trends and determine the strength of a price trend.

The window size we’ve selected for both Manual and Strategy Learner is a 7 Day window. This is to ensure we have a clear enough picture of the momentum of the stock for a given period to determine the most recent trend to determine a BUY or SELL.

# 3 MANUAL STRATEGY

## 3.1 DESCRIPTION

For Manual Strategy utilizing the three indicators listed above and determine if the user should BUY, SELL, HOLD. By combining these indicators and setting specific conditions to determine our strategy, the manual strategy can attempt to beat the benchmark strategy. From earlier projects, we know that the benchmark strategy was to buy 1000 shares to JPM and holding.

For Bollinger Band Ratio, as mentioned earlier we understand that the bands determine how we set up BUY/SELL signals. However, this indicator would only be helpful when the price approaches +/- 2 \* STD of the mean share price.

For EMA, very similar to Bollinger Band Ratio, we want to BUY when the stock is lower than the EMA, indicating the stock is undervalued, and to SELL when the stock is higher than the EMA.

For Momentum, this indicator is crucial to help determine the direction of the share price. A positive or large number indicates the stock is trending up indicated a BUY, and a negative or a small number indicates the stock is over valued indicating a SELL signal.

Combining these three indicators allows us to understand if the EMA and Bollinger bands indicate the stock is under or over valued and we leverage the momentum to help us determine the trend of the stock. For our threshold, I believe these indicator values are ideal for BUY/SELL/HOLD.

| **Name** | **Bollinger %** | **EMA** | **Momentum** |
| --- | --- | --- | --- |
| BUY | <-0.020 | <-0.05 | \>-0.05 |
| SELL | \>0.040 | \>0.025 | \>0.10 |
| HOLD | _0.39 to -0.019_ | 0.024 to -0.04 | Less than -0.04 |

**_Table 2 —_** Signals to determine to BUY, SELL, HOLD for our three indicators.

As we look at the above table, we notice an interesting situation with momentum. For both BUY and SELL, the optimized strategy is looking at increasing momentum. My understanding is that for a BUY Signal, we are looking to see if the momentum is slowing down (smaller value) to indicate the share prices is hitting the bottom and for the SELL signal, we are expecting that the share price’s momentum is high therefore we anticipate the share pricing hitting a ceiling for an eventual reversal.

By taking these signals we discretize the signal in to a -1, 0 or 1. If indicator triggers a BUY, we convert that signal to a 1. If the indicator triggers a SELL, we convert that signal to a -1. For all other scenarios we HOLD the stock, as we determine that there is not enough information to make a trade, which is converted to a 0.

This process is incredibly valuable as we can optimize our stock trading strategy based off indicator thresholds, and ultimately maximize our portfolio returns to out perform the benchmark strategy. By using the above indicators, we can establish if a trend if the stock is increasing or decreasing by the optimized thresholds of our indicators.

## 3.2 PERFORMANCE

For our in-sample data set (January 1, 2008 to December 31, 2009), we compare the Manual Strategy as discussed above and the Benchmark (Buying 1000 shares of JPM Day 1 and holding).

For our Manual Strategy, as seen below in Figure 1, the vertical blue lines indicate a LONG position, and the vertical black lines indicate a SHORT position. Our in-sample Manual Strategy outperforms the benchmark by a significant margin.


![Figure1](https://github.com/dandia89/personal-projects/blob/master/DecisionTree/Figure1_MS_In.png)

**_Figure 1—_**In-Sample Manual Strategy and Benchmark performance normalized for JPM Stock Price (Jan 1 2008 to Dec 31, 2009)

For our out-of-sample data (January 1 2010 to December 31, 2011) in Figure 2, we notice a marginal increase to our benchmark with less trades than our in-sample results. This also resulted a less optimal return.

![Figure2](https://github.com/dandia89/personal-projects/blob/master/DecisionTree/Figure2_MS_Out.png)

**_Figure 2—_** Out-of-Sample Manual Strategy and Benchmark performance normalized for JPM Stock Price (Jan 1 2010 to Dec 31, 2011)

In Table 2 below, with our numerical results, it is clear that both in-sample and out-of-sample results out perform the benchmark. These results are not surprising as tuning the parameters to optimized the in-sample data may have slightly overfit the results to our in-sample data. However, it still indicates our strategy is still effective for out-of-sample dataset.

| **Name** | **Cumulative Return** | **Standard  <br>Deviation** | **Mean  <br>Daily Returns** |
| --- | --- | --- | --- |
| Manual Strategy (In-Sample) | _0.486063_ | 0.012951 | 0.00087 |
| Benchmark (In-sample) | _0.012325_ | 0.016977 | 0.00012 |
| Manual Strategy (Out-of-Sample) | _0.063285_ | 0.008044 | 0.000154 |
| Benchmark (Out-of-Sample) | _\-0.083579_ | 0.008509 | 0.00021 |

**_Table 2 —_** Manual Strategy vs Benchmark for In-sample and Out-of-Sample

I believe the reason the Manual Strategy suceeded so well with the in-sample dataset is due to the large variance of stock prices, as it’s much more volatile therefore we can develop stronger trends with our indicators to determine BUY and SELL opportunities.

We can see that with our standard deviation numbers, for the benchmark the in-sample standard deviation was 0.016977, and our standard deviation for our out-of-sample is 0.008509. Since momentum is heavily relied on strong trends (positive or negative), this is most likely why there are less transactions for the out-of-sample data.

Also, I observed there were significant amount of trades for my in-sample approach, and with more tuning the parameters I believe I could have optimized the indicators to go LONG or SHORT for longer periods which would be optimial to reduce commission and impact costs.

# 4 STRATEGY LEARNER

## 4.1 FRAMING THE LEARNING PROBLEM

Strategy Learner is a Random Forest Learner by using a Random Tree Learner (RT) and a Bag Learner (BL). Part of the adjustment includes modifying RT and BL from a regression to a classification model. This was completed by modifying the code for each leaf from a mean calculation to a mode calculation, which will be discussed later via discretization.

Regarding the data, in-sample dates were used as training data set. We utilized all three indicator values as our “X” input. For our “Y” outputs we leverage the future stock price. Our variable N indicates how many dates we look ahead, and then we calculate the price difference from N days ahead and that would be our “Y” value from the initial date. The “Y” output threshold was developed into variables to determine if the model will BUY, SELL or HOLD our stock These variables are called YBUY and YSELL. Also, we required that the output of the classification is discretized as a -1, 0 or a 1. As mentioned earlier, that indicates a SELL, HOLD, or BUY.

Once the training data was developed, we randomized the data for each random forest learner, as that was required to aid in cross validation to ensure our model was robust enough to handle out-of-sample scenarios to build our decision tree model.

## 4.2 HYPERPARAMETERS

By several experiments, I determined that the leaf size was optimized by 5, anything larger it leads to underfitting. For the number of bags, I experiment from 20 up to 50, and determined that 50 bags will allow for the model to have the consistent high returns for in-sample and out-of-sample. The N days variable (look ahead) was set to 5 days. I experimented and found that there were too many incorrect trades with a look ahead value too far in advance.

For our YBUY and YSELL parameters, I determined YBUY was 0.045 + impact/10 and YSELL was -0.030 + impact/10. The YBUY threshold was determined based to see a positive trend in the look ahead value, indicating an upward trend. The YBUY threshold observed if the stock is truly decreasing enough for a BUY signal or if it’s a minor drop in price. The YSELL could be optimized to minimize the “wait and see” approach, however as we review the results below, it may be a marginal improvement in portfolio results.

If the Y value is greater than YBUY , we discretized a BUY signal as a 1. If our Y value is less than YSELL we discretize the signal as a -1. For conditions that are outside those hyperparameters we indicate a HOLD or a 0 discretized value.

The reason we discretized these values with YBUY and YSELL into (1,0,-1) so we could leverage the same trades function from Manual Strategy to develop a portfolio value from marketismcode without modifying the code. Also, leveraging the impact ratio to help understand the performance/trade amounts for Experiment 2, which will be discussed later in this report.

# 5 EXPERIMENT 1

Looking at Figure 3 and Figure 4, both in-sample and out-of-sample out perform the Benchmark strategy, which aligns with our hypothesis. For the Manual and Strategy Learner we tweaked the parameters for our in-sample performance, therefore the model is inherently overfit for this data set and the stock symbol JPM. This is observed as we utilize our model for our out-of-sample data with marginal returns in comparison. With significant experimentation, I believe this is the unfortunate reality of stock trading. The assumption of these models is that historical data is indicative of future stock price, which is not always a reality. For our Strategy if the learner is trained on in-sample data, it will inherently out-perform a Manual Strategy due to the number of iterations of the machine learning algorithm and cross validation. Since the Strategy Learner was tuned for in-sample data, it underperformed against the Manual Strategy for out-of-sample. From my understanding it indicates the Strategy was much more overfitted for in-sample relative to the Manual Strategy, therefore more tuning would be required.

![Figure3](https://github.com/dandia89/personal-projects/blob/master/DecisionTree/Figure3_MS_SL_In_report.png))

**_Figure 3—_** In-Sample Manual Strategy, Strategy Learner and Benchmark performance normalized for JPM Stock Price (Jan 1 2008 to Dec 31, 2009)

[![Figure4])](https://github.com/dandia89/personal-projects/blob/master/DecisionTree/Figure4_MS_SL_Out.png)
**_Figure 4—_** Out-of-Sample Manual Strategy, Strategy Learner, and Benchmark performance normalized for JPM Stock Price (Jan 1 2008 to Dec 31, 2009)

# 6 EXPERIMENT 2

For Experiment 2, we utilize 0.0, 0.05 and 0.10 impact to understand the changes of our Strategy Learner from two metrics: amount of trades and portfolio values.

Since our YBUY and YSELL hyperparameters are developed based off impact, and our marketismcode modifies our portfolio returns for each trade, an increase of impact should theoretically change our outcome. As shown in Figure 5, as impact is increased, our amount of trades decrease. This is mostly due to the threshold of YBUY and YSELL are modified by impact/10.

For Figure 6, we notice that an impact has a significant change on our portfolio returns, anything greater than 0.0, our portfolio returns would be negative. This is due to the hyperparameters of YBUY and YSELL are not optimized to an impact of 0, and the portfolio returns are changed significantly due to the amount of trades observed. We are ranging 105-70 trades in two years, which is why our returns are reduced.

![Figure5](https://github.com/dandia89/personal-projects/blob/master/DecisionTree/Figure5_Impact_trades.png)
**_Figure 5—_**Number of trades observed for Impact 0.0, 0.05 and 0.10 utilizing Strategy Learner for JPM Stock Price (Jan 1 2008 to Dec 31, 2009)

![Figure6](https://github.com/dandia89/personal-projects/blob/master/DecisionTree/Figure6_Impact_values.png)

**_Figure 6—_** Normalized Portfolio Values for Impact 0.0, 0.05 and 0.10 utilizing Strategy Learner for JPM Stock Price (Jan 1 2008 to Dec 31, 2009)

# 7 REFERENCES

1. James Chen (February 23, 2024) What is EMA? How to Use Exponential Moving Average With Formula. Sourced on March 9, 2024 from <https://www.investopedia.com/terms/e/ema.asp>
