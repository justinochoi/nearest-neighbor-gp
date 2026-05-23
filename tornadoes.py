import polars as pl 
from nngp_functions import * 

df = pl.read_csv('/Users/justinchoi/Downloads/1950-2024_actual_tornadoes.csv')

df_filtered = (
    df.filter(pl.col('yr') >= 2007, pl.col('mag') != -9)
    .select(['yr','mo','dy','slon','slat','mag'])
)

