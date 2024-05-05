import os
from nn_model import ThreeLayerNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import seaborn as sns
df=pd.read_csv('hyperparameter_search_results.csv')
filtered_df = df[df['lr'] == 0.005]

# 构建热力图需要的透视表
pivot_table = filtered_df.pivot(index='reg', columns='hidden_size', values='val_acc')

# 使用Seaborn绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Wistia")
plt.title('Validation Accuracy for Different Regularization and Hidden Sizes')
plt.ylabel('Regularization λ')
plt.xlabel('Hidden Layer Size')
#plt.savefig('Validation Accuracy for Different Regularization and Hidden Sizes.png')
plt.show()