import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# plot newCIM vs old CIM (no PCA)
new_CIM_2000_path = './new_CIM_ISOLET_result/Accuracy_between_level2000_newCIM.csv'
CIM_2000_path = './isolet_result/Accuracy_between_level2000.csv'
new_CIM_5000_path = './new_CIM_ISOLET_result/Accuracy_between_level5000_newCIM.csv'
CIM_5000_path = './isolet_result/Accuracy_between_level5000.csv'
new_CIM_10000_path = './new_CIM_ISOLET_result/Accuracy_between_level10000_newCIM.csv'
CIM_10000_path = './isolet_result/Accuracy_between_level.csv'

df = pd.read_csv(new_CIM_2000_path)
new_CIM_2000 = np.array(df.iloc[:, 1])
df = pd.read_csv(CIM_2000_path)
CIM_2000 = np.array(df.iloc[:, 1])


df = pd.read_csv(new_CIM_5000_path)
new_CIM_5000 = np.array(df.iloc[:, 1])
df = pd.read_csv(CIM_5000_path)
CIM_5000 = np.array(df.iloc[:, 1])

df = pd.read_csv(new_CIM_10000_path)
new_CIM_10000 = np.array(df.iloc[:, 1])
df = pd.read_csv(CIM_10000_path)
CIM_10000 = np.array(df.iloc[:, 1:])
CIM_10000 = np.average(CIM_10000, axis=1)

# 畫newCIM vs old CIM (without PCA)
plt.plot(np.arange(2, 22), new_CIM_2000, 'bo-', label='new Dimension 2000')
plt.plot(np.arange(2, 22), CIM_2000, color='tab:blue',
         linestyle='-', marker='x', label='CIM Dimension 2000')
plt.plot(np.arange(2, 22), new_CIM_5000, 'ko-', label='new Dimension 5000')
plt.plot(np.arange(2, 22), CIM_5000, color='tab:gray',
         linestyle='-', marker='x', label='CIM Dimension 5000')
plt.plot(np.arange(2, 22), new_CIM_10000, 'ro-', label='new Dimension 10000')
plt.plot(np.arange(2, 22), CIM_10000, color='tab:red',
         linestyle='-', marker='x', label='CIM Dimension 10000')
plt.title('Accuracy between different CIM level(no PCA)')
plt.grid()
plt.xlabel('Level')
plt.xticks(np.arange(2, 22))
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('./new_CIM_ISOLET_result/oldCIM_vs_newCIM_noPCA.png')
plt.close()


# plot newCIM+PCA vs old CIM +PCA

new_CIM_2000_PCA_path = './new_CIM_ISOLET_result/Accuracy_between_level2000_newCIM_PCA.csv'
CIM_2000_PCA_path = './new_CIM_ISOLET_result/Accuracy_between_level2000_oldCIM_PCA.csv'
new_CIM_5000_PCA_path = './new_CIM_ISOLET_result/Accuracy_between_level5000_newCIM_PCA.csv'
CIM_5000_PCA_path = './new_CIM_ISOLET_result/Accuracy_between_level5000_oldCIM_PCA.csv'
new_CIM_10000_PCA_path = './new_CIM_ISOLET_result/Accuracy_between_level10000_newCIM_PCA.csv'
CIM_10000_PCA_path = './new_CIM_ISOLET_result/Accuracy_between_level10000_oldCIM_PCA.csv'

# load the result
df = pd.read_csv(new_CIM_2000_PCA_path)
new_CIM_2000_PCA = np.array(df.iloc[:, 1])
df = pd.read_csv(CIM_2000_PCA_path)
CIM_2000_PCA = np.array(df.iloc[:, 1])


df = pd.read_csv(new_CIM_5000_PCA_path)
new_CIM_5000_PCA = np.array(df.iloc[:, 1])
df = pd.read_csv(CIM_5000_PCA_path)
CIM_5000_PCA = np.array(df.iloc[:, 1])

df = pd.read_csv(new_CIM_10000_PCA_path)
new_CIM_10000_PCA = np.array(df.iloc[:, 1])
df = pd.read_csv(CIM_10000_PCA_path)
CIM_10000_PCA = np.array(df.iloc[:, 1:])


# 畫newCIM vs old CIM (without PCA)
plt.plot(np.arange(2, 22), new_CIM_2000_PCA, 'bo-', label='new+PCA Dim 2000')
plt.plot(np.arange(2, 22), CIM_2000_PCA, color='tab:blue',
         linestyle='-', marker='x', label='CIM+PCA Dim 2000')
plt.plot(np.arange(2, 22), new_CIM_5000_PCA, 'ko-', label='new+PCA Dim 5000')
plt.plot(np.arange(2, 22), CIM_5000_PCA, color='tab:gray',
         linestyle='-', marker='x', label='CIM+PCA Dim 5000')
plt.plot(np.arange(2, 22), new_CIM_10000_PCA, 'ro-', label='new+PCA Dim 10000')
plt.plot(np.arange(2, 22), CIM_10000_PCA, color='tab:red',
         linestyle='-', marker='x', label='CIM+PCA Dim 10000')
plt.title('CIM+PCA vs new CIM+PCA')
plt.grid()
plt.xlabel('Level')
plt.xticks(np.arange(2, 22))
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('./new_CIM_ISOLET_result/CIM+PCA_vs_newCIM+PCA.png')
plt.close()


# plot newCIM vs newCIM+PCA

plt.plot(np.arange(2, 22), new_CIM_2000_PCA, 'bo-', label='new+PCA Dim 2000')
plt.plot(np.arange(2, 22), new_CIM_2000, color='tab:blue',
         linestyle='-', marker='x', label='new Dim 2000')
plt.plot(np.arange(2, 22), new_CIM_5000_PCA, 'ko-', label='new+PCA Dim 5000')
plt.plot(np.arange(2, 22), new_CIM_5000, color='tab:gray',
         linestyle='-', marker='x', label='new Dim 5000')
plt.plot(np.arange(2, 22), new_CIM_10000_PCA, 'ro-', label='new+PCA Dim 10000')
plt.plot(np.arange(2, 22), new_CIM_10000, color='tab:red',
         linestyle='-', marker='x', label='new Dim 10000')
plt.title('newCIM+PCA vs newCIM')
plt.grid()
plt.xlabel('Level')
plt.xticks(np.arange(2, 22))
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('./new_CIM_ISOLET_result/newCIM+PCA_vs_newCIM.png')
plt.close()

# plot old CIM vs oldCIM+PCA

plt.plot(np.arange(2, 22), CIM_2000_PCA, 'bo-', label='CIM+PCA Dim 2000')
plt.plot(np.arange(2, 22), CIM_2000, color='tab:blue',
         linestyle='-', marker='x', label='CIM Dim 2000')
plt.plot(np.arange(2, 22), CIM_5000_PCA, 'ko-', label='CIM+PCA Dim 5000')
plt.plot(np.arange(2, 22), CIM_5000, color='tab:gray',
         linestyle='-', marker='x', label='CIM Dim 5000')
plt.plot(np.arange(2, 22), CIM_10000_PCA, 'ro-', label='CIM+PCA Dim 10000')
plt.plot(np.arange(2, 22), CIM_10000, color='tab:red',
         linestyle='-', marker='x', label='CIM Dim 10000')
plt.title('CIM+PCA vs CIM')
plt.grid()
plt.xlabel('Level')
plt.xticks(np.arange(2, 22))
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('./new_CIM_ISOLET_result/CIM+PCA_vs_CIM.png')
plt.close()

# plot all the results
# Dim=2000
plt.plot(np.arange(2, 22), CIM_2000_PCA, 'bx-', label='CIM+PCA Dim 2000')
plt.plot(np.arange(2, 22), CIM_2000, color='tab:blue',
         linestyle='-', marker='x', label='CIM Dim 2000')
plt.plot(np.arange(2, 22), new_CIM_2000_PCA, 'ro-', label='new+PCA Dim 2000')
plt.plot(np.arange(2, 22), new_CIM_2000, color='tab:red',
         linestyle='-', marker='o', label='new Dim 2000')

plt.title('Dimension 2000')
plt.grid()
plt.xlabel('Level')
plt.xticks(np.arange(2, 22))
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('./new_CIM_ISOLET_result/All_results_Dim2000.png')
plt.close()
# Dim=5000
plt.plot(np.arange(2, 22), CIM_5000_PCA, 'bx-', label='CIM+PCA Dim 5000')
plt.plot(np.arange(2, 22), CIM_5000, color='tab:blue',
         linestyle='-', marker='x', label='CIM Dim 5000')
plt.plot(np.arange(2, 22), new_CIM_5000_PCA, 'ro-', label='new+PCA Dim 5000')
plt.plot(np.arange(2, 22), new_CIM_5000, color='tab:red',
         linestyle='-', marker='o', label='new Dim 5000')

plt.title('Dimension 5000')
plt.grid()
plt.xlabel('Level')
plt.xticks(np.arange(2, 22))
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('./new_CIM_ISOLET_result/All_results_Dim5000.png')
plt.close()
# Dim=10000
plt.plot(np.arange(2, 22), CIM_10000_PCA, 'bx-', label='CIM+PCA Dim 10000')
plt.plot(np.arange(2, 22), CIM_10000, color='tab:blue',
         linestyle='-', marker='x', label='CIM Dim 10000')
plt.plot(np.arange(2, 22), new_CIM_10000_PCA, 'ro-', label='new+PCA Dim 10000')
plt.plot(np.arange(2, 22), new_CIM_10000, color='tab:red',
         linestyle='-', marker='o', label='new Dim 10000')

plt.title('Dimension 10000')
plt.grid()
plt.xlabel('Level')
plt.xticks(np.arange(2, 22))
plt.ylabel('Accuracy(%)')
plt.legend()
plt.savefig('./new_CIM_ISOLET_result/All_results_Dim10000.png')
plt.close()
