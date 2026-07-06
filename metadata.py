import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd_meta = pd.read_csv('metadata.csv', index_col=0)
energy_lst = ['electricity', 'steam', 'chilledwater']
pd_meta_selected = pd.DataFrame()

for energy in energy_lst:
    pd_energy = pd.read_csv(f'{energy}_selected.csv', index_col=0, parse_dates=True)
    # pd_energy = pd.read_csv(f'{energy}_continual.csv', index_col=0, parse_dates=True)    # For the building meta in continual learning
    building_lst = pd_energy.columns

    pd_ = pd_meta.loc[building_lst,['primaryspaceusage', 'sub_primaryspaceusage', 'sqm']]
    pd_['energy'] = energy
    pd_meta_selected = pd.concat([pd_meta_selected, pd_], axis=0)
    
pd_meta_selected.to_csv('meta_selected.csv')
# pd_meta_selected.to_csv('meta_continual.csv')    # For the building meta in continual learning

'''
pd_meta = pd.concat([meta_encoded, pd_feature], axis=1)
pd_meta_test = pd.concat([meta_encoded, pd_feature_test], axis=1)

np_train_meta = np.zeros((len(building_lst), len(pd_meta.columns)))
np_test_meta = np.zeros((len(building_lst), len(pd_meta_test.columns)))

for i, building in enumerate(building_lst):
    train_meta = pd_meta.loc[building,:].values
    np_train_meta[i] = train_meta

    test_meta = pd_meta_test.loc[building, :].values
    np_test_meta[i] = test_meta

np_train_meta[np.isnan(np_train_meta)] = 0
np_test_meta[np.isnan(np_test_meta)] = 0
'''
# np_test_meta = (np_test_meta - np.min(np_train_meta)) / (np.max(np_train_meta) - np.min(np_train_meta))
# np_train_meta = (np_train_meta - np.min(np_train_meta)) / (np.max(np_train_meta) - np.min(np_train_meta))
# np_train_meta = np_train_meta.reshape((np_train_meta.shape[0], np_train_meta.shape[1], 1))

#np.save('meta_train.npy', np_train_meta)
#np.save('meta_test.npy', np_test_meta)
