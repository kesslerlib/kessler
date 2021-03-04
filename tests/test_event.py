import kessler
from kessler import EventDataset
import pandas as pd
df=pd.read_csv('sample.csv')
EventDataset.from_pandas(df)
