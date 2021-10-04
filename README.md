# Taichung DE Plant PV Forecast  

Data source: Taichung DE Plant, CWB WUQI weather data.

Period: 2016/01/01 - 2017/12/31

Frequency : 1 hour.

# Data Info

Power data :
             
                 power(KWH) : PV plant produce energy ( 𝐾𝑊𝐻 ).

                 Solar Irradiance : power per unit area received from the sun ( 𝐾𝐽/𝑚2 )

Weather data :
                
                StnPres : station pressure ( ℎ𝑃𝑎 ).

                SeaPres : sea surface pressure ( ℎ𝑃𝑎 ).

                Temperature : temperature ( ∘𝐶 ).

                Td dew point : dew point temperature( ∘𝐶 ).

                RH : relative humidity ( % ).

                WS : wind speed( 𝑚/𝑠 ).

                WD : wind direction ( ∘ ).

                WSGust : the maximum gust wind speed( 𝑚/𝑠 ).

                WDGust : the maximum gust wind direction ( ∘ ).

                Precp : precipitation ( 𝑚𝑚 ).

                PrecpHour : precipitation duration( 𝐻𝑜𝑢𝑟 ).

                SunShine : sunshine duration( 𝐻𝑜𝑢𝑟 ).

                GlobRad : global radiation( 𝑀𝐽/𝑚2 ).

                Visb : visibility.

                UVI : This station can't collect UVI.

                Cloud Amount : refers to the fraction of the sky obscured by clouds.


# Model Score 

predict value : $ \hat{y}$  

actual value : $y$  

max of y : $y_{max}$

RMSE (root mean squared error) :

$$\sqrt{ \frac{1}{n} \sum_{k=1}^n ({y_{k} -\hat{y_{k}}})^2 }$$

MAE (mean absoluate error) :

$$ \frac{1}{n} \sum_{k=1}^n \vert {y_{k} -\hat{y_{k}}} \vert $$

MRE (mean relative error) :

$$ \frac{100\%}{n} \sum_{k=1}^n \frac{{y_{k} -\hat{y_{k}}}}{y_{max}}  $$

$ R^{2} $   (coefficient of determination) :

$$ R^{2} \equiv 1 - \frac {SS_{res}}{SS_{tot}}$$

$$SS_{res} = \sum_{k=1}^n  { (\hat{y_{k}} - \tilde {y})^2}   ,SS_{tot} = \sum_{k=1}^n  { ({y_{k}} - \tilde {y})^2} 
, \tilde {y} = \frac{1}{n} \sum_{k=1}^n {y_{k}} $$


Score   | SARIMA   | SVR      | Xgboost   | Neural Network| LSTM
--------|---------:|---------:| ---------:|---------:|--------
RMSE    | 101.0158 |  90.5111 | 98.5680   | 98.3961  |85.7867
MAE     | 47.0276  |  41.3922 |  46.5779  | 56.2184  |38.3409
MRE     | 3.28%    | 2.88%    |  3.25%    | 3.92%    |2.67%
$R^{2}$ | 0.9002   | 0.9338   |  0.9215   | 0.9218   |0.9406





