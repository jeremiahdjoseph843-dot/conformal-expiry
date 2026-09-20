import pyarrow.parquet as pq, pyarrow as pa, pandas as pd, glob, re
from datetime import datetime
frames=[]
for f in sorted(glob.glob('data/arxiv/data/*.parquet')):
    t=pq.read_table(f, columns=['id','title','abstract','categories','versions'])
    df=t.to_pandas()
    df['primary']=df['categories'].str.split(' ').str[0]
    df=df[df['primary'].str.startswith('cs.')]
    df['v1']=df['versions'].apply(lambda v: v[0]['created'] if len(v) else None)
    df=df.drop(columns=['versions','categories'])
    frames.append(df); print(f, len(df), flush=True)
df=pd.concat(frames)
df['date']=pd.to_datetime(df['v1'], format='%a, %d %b %Y %H:%M:%S GMT', errors='coerce')
df=df.dropna(subset=['date'])
df['year']=df['date'].dt.year
print(len(df)); print(df['year'].value_counts().sort_index())
print(df['primary'].value_counts().head(25))
df.to_parquet('data/cs_all.parquet')
