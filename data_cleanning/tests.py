import numpy as np
import pandas as pd
a = np.array([1, 2, 3, 4]).tolist()
b = ','.join(str(x) for x in a)
# t = pd.DataFrame(data=[a, b], columns='test')
# print(t)
df = pd.DataFrame([a, b], columns=list('AB'))
print(df)

