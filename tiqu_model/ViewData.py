import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取文件
tiqu = pd.read_csv('./Data/tiqu.csv')
fig1 = plt.figure()
sns.regplot(x='x2',y='y2',data=tiqu)
fig1.show()

fig2 = plt.figure()
sns.regplot(x='x2',y='y1',data=tiqu)
fig2.show()

fig3 = plt.figure()
sns.regplot(x='x4',y='y1',data=tiqu)
fig3.show()

#

sns.lmplot(x='x2',y='y2',data=tiqu,hue='Num',markers=['*','o','+'])
plt.show()