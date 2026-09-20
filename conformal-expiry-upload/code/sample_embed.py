import pandas as pd, numpy as np, time
from sentence_transformers import SentenceTransformer
CATS=['cs.CV','cs.LG','cs.CL','cs.AI','cs.RO','cs.IT','cs.CR','cs.SE','cs.HC','cs.NI','cs.DS','cs.DC']
df=pd.read_parquet('data/cs_all.parquet')
df=df[df['primary'].isin(CATS)]
df=df[(df['date']>='2013-01-01')&(df['date']<'2026-09-01')]
df['month']=df['date'].dt.to_period('M').astype(str)
rng=np.random.RandomState(42)
parts=[]
for m,g in df.groupby('month'):
    n=min(600,len(g)); parts.append(g.sample(n, random_state=rng))
s=pd.concat(parts).sort_values('date').reset_index(drop=True)
print(len(s), s['month'].nunique(), flush=True)
s['text']=(s['title'].str.replace('\n',' ')+'. '+s['abstract'].str.replace('\n',' ')).str.strip()
s[['id','primary','date','month','text']].to_parquet('data/sample.parquet')
model=SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); model.max_seq_length=256
t=time.time()
E=model.encode(s['text'].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
np.save('data/emb_minilm.npy', E.astype(np.float32))
print('done', E.shape, time.time()-t, flush=True)
