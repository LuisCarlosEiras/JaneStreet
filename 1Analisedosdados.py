# Análise dos dados

import dask.dataframe as dd
import dask
import pandas as pd
import numpy as np
import gc

# Define o caminho dos dados
data_path = "/content/gdrive/My Drive/Colab Notebooks/jane-street-real-time-market-data-forecasting/"

# Lê o arquivo train.parquet utilizando Dask
train_df = dd.read_parquet(f"{data_path}train.parquet", engine='pyarrow')

# Lê o arquivo features.csv utilizando Dask
features_df = dd.read_csv(f"{data_path}features.csv")

# Lê o arquivo responders.csv utilizando Dask
responders_df = dd.read_csv(f"{data_path}responders.csv")

# Análise 1: Contagem de valores nulos por coluna
null_counts = train_df.isnull().sum().compute()
print("Contagem de valores nulos por coluna:")
print(null_counts)

# Análise 2: Estatísticas descritivas para as colunas numéricas
numeric_stats = train_df.describe().compute()
print("Estatísticas descritivas:")
print(numeric_stats)

# Análise 3: Correlação entre as features numéricas
correlation_matrix = train_df.corr().compute()
print("Matriz de Correlação:")
print(correlation_matrix)

'''
Contagem de valores nulos por coluna:
date_id               0
time_id               0
symbol_id             0
weight                0
feature_00      3182052
                 ...   
responder_5           0
responder_6           0
responder_7           0
responder_8           0
partition_id          0
Length: 93, dtype: int64
Estatísticas descritivas:
            date_id       time_id     symbol_id        weight    feature_00  \
count  4.712734e+07  4.712734e+07  4.712734e+07  4.712734e+07  4.394529e+07   
mean   1.005479e+03  4.687057e+02  1.810239e+01  2.009445e+00  5.738326e-01   
std    4.451819e+02  2.725187e+02  1.130165e+01  1.129388e+00  1.327413e+00   
min    0.000000e+00  0.000000e+00  0.000000e+00  1.499667e-01 -5.794129e+00   
25%    6.770000e+02  2.580000e+02  1.100000e+01  1.467334e+00 -3.038115e-02   
50%    1.059000e+03  5.090000e+02  1.900000e+01  2.085095e+00  8.484703e-01   
75%    1.374000e+03  7.480000e+02  2.900000e+01  4.103159e+00  2.173780e+00   
max    1.698000e+03  9.670000e+02  3.800000e+01  1.024042e+01  6.477002e+00   

         feature_01    feature_02    feature_03    feature_04    feature_05  \
count  4.394529e+07  4.394529e+07  4.394529e+07  4.394529e+07  4.712734e+07   
mean   1.019200e-02  5.731025e-01  5.727449e-01 -8.522294e-04 -3.837982e-02   
std    1.084940e+00  1.322870e+00  1.322849e+00  1.037188e+00  1.020120e+00   
min   -5.741592e+00 -5.726010e+00 -5.601890e+00 -5.799880e+00 -2.535040e+01   
25%   -3.460638e-01 -1.580641e-02 -1.959961e-02 -4.667797e-01 -3.245846e-01   
50%    3.486357e-01  8.425343e-01  8.444433e-01  1.134397e-01  1.158459e-01   
75%    1.405966e+00  2.192164e+00  2.148272e+00  1.145805e+00  1.027239e+00   
max    6.292391e+00  6.490265e+00  6.695623e+00  6.164160e+00  3.572622e+01   

       ...    feature_78   responder_0   responder_1   responder_2  \
count  ...  4.710730e+07  4.712734e+07  4.712734e+07  4.712734e+07   
mean   ... -1.255889e-02 -1.545149e-03 -8.656001e-04 -1.419604e-04   
std    ...  9.435917e-01  5.912108e-01  5.875515e-01  5.991786e-01   
min    ... -6.159515e+00 -5.000000e+00 -5.000000e+00 -5.000000e+00   
25%    ... -2.463093e-01 -1.319363e-01 -1.077630e-01 -1.124551e-01   
50%    ... -1.093368e-01  4.871721e-03  1.293741e-02  1.847853e-03   
75%    ...  3.884036e-01  4.061097e-01  3.367091e-01  4.102618e-01   
max    ...  2.574540e+02  5.000000e+00  5.000000e+00  5.000000e+00   

        responder_3   responder_4   responder_5   responder_6   responder_7  \
count  4.712734e+07  4.712734e+07  4.712734e+07  4.712734e+07  4.712734e+07   
mean  -1.638009e-02 -1.244198e-02 -1.662421e-02 -2.140649e-03  1.476110e-03   
std    8.192309e-01  8.728399e-01  7.345458e-01  8.898516e-01  9.160423e-01   
min   -5.000000e+00 -5.000000e+00 -5.000000e+00 -5.000000e+00 -5.000000e+00   
25%   -2.097922e-01 -2.580736e-01 -1.559837e-01 -2.657704e-01 -2.742171e-01   
50%    1.666263e-02  1.674504e-02  1.926651e-02 -3.626317e-03 -4.533414e-03   
75%    4.474130e-01  5.368025e-01  3.394354e-01  5.354323e-01  5.595564e-01   
max    5.000000e+00  5.000000e+00  5.000000e+00  5.000000e+00  5.000000e+00   

        responder_8  
count  4.712734e+07  
mean  -1.113621e-03  
std    8.644118e-01  
min   -5.000000e+00  
25%   -2.373489e-01  
50%    2.794336e-03  
75%    5.224141e-01  
max    5.000000e+00  

[8 rows x 92 columns]
Matriz de Correlação:
               date_id   time_id  symbol_id    weight  feature_00  feature_01  \
date_id       1.000000  0.075434   0.121931  0.208156    0.393219    0.013441   
time_id       0.075434  1.000000   0.012671  0.013284   -0.005658   -0.328443   
symbol_id     0.121931  0.012671   1.000000 -0.216795    0.005386   -0.000836   
weight        0.208156  0.013284  -0.216795  1.000000    0.078567    0.010653   
feature_00    0.393219 -0.005658   0.005386  0.078567    1.000000    0.061294   
...                ...       ...        ...       ...         ...         ...   
responder_5  -0.021899 -0.000662  -0.001269 -0.011854   -0.043454   -0.010140   
responder_6   0.000926 -0.011759   0.000669 -0.001494    0.004444   -0.011560   
responder_7   0.004339 -0.018347   0.002048 -0.007106    0.008789   -0.006881   
responder_8   0.000386 -0.005567   0.000172 -0.000468    0.002317   -0.008295   
partition_id  0.993890  0.075792   0.121791  0.211616    0.396448    0.013876   

              feature_02  feature_03  feature_04  feature_05  ...  \
date_id         0.392623    0.392672    0.003384   -0.032270  ...   
time_id         0.020534    0.015552   -0.112057    0.010793  ...   
symbol_id       0.005309    0.005314    0.000303   -0.004678  ...   
weight          0.077878    0.077973    0.002208   -0.000275  ...   
feature_00      0.943672    0.945736    0.034080   -0.038501  ...   
...                  ...         ...         ...         ...  ...   
responder_5    -0.042443   -0.043441   -0.024132   -0.017854  ...   
responder_6     0.006692    0.005205   -0.031607   -0.016263  ...   
responder_7     0.010205    0.009448   -0.016496   -0.009792  ...   
responder_8     0.004096    0.002805   -0.029401   -0.025364  ...   
partition_id    0.395601    0.395688    0.003331   -0.033693  ...   

              responder_0  responder_1  responder_2  responder_3  responder_4  \
date_id         -0.003219    -0.005199    -0.000420    -0.019114    -0.014120   
time_id          0.028375     0.095801     0.009457     0.001533     0.011352   
symbol_id       -0.000377    -0.000334    -0.000139    -0.000816     0.001190   
weight           0.000354     0.007369    -0.000014    -0.010330    -0.012929   
feature_00      -0.002139    -0.000457    -0.000935    -0.035421    -0.019288   
...                   ...          ...          ...          ...          ...   
responder_5      0.137676     0.070281     0.296056     0.577362     0.322944   
responder_6     -0.119742    -0.054210    -0.055690     0.727253     0.355831   
responder_7     -0.043369    -0.095585    -0.024204     0.340227     0.802048   
responder_8     -0.060355    -0.032970    -0.149800     0.328320     0.164291   
partition_id    -0.003130    -0.004647    -0.000430    -0.018795    -0.014142   

              responder_5  responder_6  responder_7  responder_8  partition_id  
date_id         -0.021899     0.000926     0.004339     0.000386      0.993890  
time_id         -0.000662    -0.011759    -0.018347    -0.005567      0.075792  
symbol_id       -0.001269     0.000669     0.002048     0.000172      0.121791  
weight          -0.011854    -0.001494    -0.007106    -0.000468      0.211616  
feature_00      -0.043454     0.004444     0.008789     0.002317      0.396448  
...                   ...          ...          ...          ...           ...  
responder_5      1.000000     0.299872     0.145044     0.621764     -0.021281  
responder_6      0.299872     1.000000     0.431704     0.446989      0.000759  
responder_7      0.145044     0.431704     1.000000     0.205747      0.003854  
responder_8      0.621764     0.446989     0.205747     1.000000      0.000348  
partition_id    -0.021281     0.000759     0.003854     0.000348      1.000000  
'''