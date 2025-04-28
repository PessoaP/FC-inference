# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import fcsparser
from matplotlib.lines import Line2D

np.random.seed(42)

# %%
AHratio = lambda df: (df["FSC-A"]/df["FSC-H"]).to_numpy()
check_AH = lambda df: np.abs(AHratio(df)-1)<.3

def acceptable_data(file,cols=['FSC-A','FL1-A'],AHratio_clean = False):
    datafile = fcsparser.parse(file, reformat_meta=True)[-1]

    df_array = datafile.to_numpy()
    removal =  np.any(np.isnan(df_array),axis=1) #marks for removal

    if AHratio_clean:
        removal += check_AH(df_array)

    df_array = datafile[cols].to_numpy()
    removal += np.any(df_array<=0,axis=1) # remove nans and non-positive

    return datafile[np.logical_not(removal)]

def take_log(df,base=10):
    ans = pd.DataFrame()
    for ind in df.columns:
        ans['log '+ind] = np.log(df[ind].to_numpy())/np.log(base)
    return ans

# %%
sac_files_d1 = ['ECOLandSACatDIL_368rpm_sequentialstressContinousCulture/'+st+'.fcs' for st in ['A01 R3D1T1','A02 R3D1T2','A03 R3D1T3']]
sac_files_d2 = ['ECOLandSACatDIL_368rpm_sequentialstressContinousCulture/'+st+'.fcs' for st in ['A05 R3D2T1','A06 R3D2T2','A07 R3D2T3']]
sac_files_d3 = ['ECOLandSACatDIL_368rpm_sequentialstressContinousCulture/'+st+'.fcs' for st in ['A08 R3D3T1','A09 R3D3T2','A10 R3D3T3']]
stat_phase = ['ScerevisiaeOldFC/A0{}.fcs'.format(i) for i in range(1,10)]

# %%
pd1 = pd.concat([acceptable_data(x) for x in (sac_files_d1)],ignore_index=True)
pd2 = pd.concat([acceptable_data(x) for x in (sac_files_d2)],ignore_index=True)
pd3 = pd.concat([acceptable_data(x) for x in (sac_files_d3)],ignore_index=True)
dils=(.12,.23,.33)

# %%
from scipy.linalg import sqrtm
from sklearn.svm import OneClassSVM



def SVM_clean(df,nu=.025,Z_train=True):
    X = take_log(df[['FL1-A','FSC-A','FSC-H']],base=10).to_numpy()
    X[:,2] = X[:,2]-X[:,1]
    Z = np.dot(X-X.mean(axis=0),
               np.linalg.inv(sqrtm(np.cov(X.T))))

    keep = np.logical_and(np.abs(X[:,2])<np.log10(1.75),
                          X[:,1]>5.25)
    test = np.arange(0,keep.sum(),3)
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)


    model.fit(Z[keep][test])
    y_pred = model.predict(Z)

    print(y_pred.mean())
    
    return model,X,Z,y_pred

# %%
indexes = [SVM_clean(df)[-1]==1 for df in (pd1,pd2,pd3)]

# %%
[(df[ind]).to_csv('complete_d={}.csv'.format(d),index=False) for (df,d,ind) in zip((pd1,pd2,pd3),dils,indexes)]

# %%
def prepgraph(ax,df,indexes,color,label=''):
    ax.scatter(df[indexes[0]],df[indexes[1]],s=6,alpha=.05,color=color,label=label)
    ax.set_xlabel(r'$\log_{10}$' + indexes[0][-4:])
    ax.set_ylabel(r'$\log_{10}$' + indexes[1][-4:])

def graphs3(ax,df,color,label):
    logdf = take_log(df[['FL1-A','FSC-A','FSC-H']])
    prepgraph(ax[0],logdf,['log FSC-A','log FL1-A'],color=color,label=label)
    prepgraph(ax[1],logdf,['log FSC-H','log FL1-A'],color=color)
    prepgraph(ax[2],logdf,['log FSC-H','log FSC-A'],color=color)

def separated_graphs(ax,df,y):
    graphs3(ax,df[~y],'b','Kept')
    graphs3(ax,df[y],'r','Removed')

legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='Removed', 
               markerfacecolor='b', markersize=10, alpha=1.0),
        Line2D([0], [0], marker='o', color='none', label='Kept', 
               markerfacecolor='r', markersize=10, alpha=1.0)
    ]

# %%
fig,ax = plt.subplots(3,3,figsize=(13,8))
titles = ('High stress','Low stress','Control')
for (i,df) in enumerate((pd1,pd2,pd3)):
    separated_graphs(ax[:,i],df,indexes[i])
    ax[0,i].set_title(r'Dilution rate: {} $h^{{-1}}$ ({})'.format(dils[i],titles[i]),fontsize=15)
ax[0,0].legend(handles=legend_elements, frameon=False)
plt.tight_layout()
plt.savefig('SI5',dpi=500)

# %%



