import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

energy_lst = ['electricity', 'steam', 'chilledwater']
location_lst = [['Fox', 'Hog'], ['Hog', 'Bull'], ['Fox', 'Bull']]

np_elec = np.zeros((24*335*54,2))
np_elec_test = np.zeros((24*181*54,2))

pd_weather = pd.read_csv('data/weather.csv', index_col=0, parse_dates=True)

def pd_time_information(Timestamp):
    pd_time_information = pd.DataFrame(columns=['month', 'day_of_week', 'day_of_month', 'hour'])
    for i in range(len(Timestamp)):
        month = int(str(pd.Timestamp(Timestamp[i]))[5:7])
        day_of_week = pd.Timestamp(Timestamp[i]).dayofweek + 1
        day_of_month = int(str(pd.Timestamp(Timestamp[i]))[8:10])
        hour = int(str(pd.Timestamp(Timestamp[i]))[11:13])
        pd_time_information.loc[i] = [month, day_of_week, day_of_month, hour]
    pd_time_information.index = Timestamp
    return pd_time_information

def filtering_by_hours(train_data, pd_time, Timestamp):
    pd_time_index = pd.to_datetime(Timestamp)
    pd_time.index = pd_time_index
    pd_time['Energy [KW]'] = train_data

    for i in range(24):
        hour = i
        energy_demand = pd_time[pd_time['hour'] == hour]['Energy [KW]'].values
        energy_demand_25 = np.percentile(energy_demand, 25)
        energy_demand_75 = np.percentile(energy_demand, 75)
        energy_demand_median = np.percentile(energy_demand, 50)
        lower_bound = energy_demand_25 - 1.5 * (energy_demand_75 - energy_demand_25)
        upper_bound = energy_demand_75 + 1.5 * (energy_demand_75 - energy_demand_25)
        for j in range(len(energy_demand)):
            if (lower_bound <= energy_demand[j] <= upper_bound) or np.isnan(energy_demand[j]):
                pass
            else:
                outlier_index = pd_time[pd_time['hour'] == hour]['Energy [KW]'].index[j]
                pd_time.loc[outlier_index, 'Energy [KW]'] = np.percentile(energy_demand, 50)
    return pd_time

for i, energy in enumerate(energy_lst):
    pd_energy = pd.read_csv(f'data/{energy}_selected.csv', index_col=0, parse_dates=True)
    pd_energy_test = pd.read_csv(f'data/{energy}_selected_test.csv', index_col=0, parse_dates=True)

    for j in range(18):
        cnt = j // 9
        np_temp = pd_weather[pd_weather['site_id']==location_lst[i][cnt]].loc['2016-02-01 00:00:00':'2016-12-31 23:00:00', 'airTemperature'].resample('60min').interpolate('linear').to_frame().values
        np_temp = np_temp.reshape((np_temp.shape[0],1))

        np_temp_test = pd_weather[pd_weather['site_id']==location_lst[i][cnt]].loc['2017-01-01 00:00:00':'2017-06-30 23:00:00', 'airTemperature'].resample('60min').interpolate('linear').to_frame().values
        np_temp_test = np_temp_test.reshape((np_temp_test.shape[0], 1))

        np_load = pd_energy.iloc[:,j].values
        np_load = np_load.reshape((np_load.shape[0],1))

        np_load_test = pd_energy_test.iloc[:, j].values
        np_load_test = np_load_test.reshape((np_load_test.shape[0], 1))

        np_ = np.hstack((np_temp, np_load))
        np_test_ = np.hstack((np_temp_test, np_load_test))

        np_elec[24*335*(j+18*i): 24*335*(j+18*i+1), :] = np_
        np_elec_test[24*181*(j+18*i): 24*181*(j+18*i+1), :] = np_test_

train_timestamp_lst = [pd.to_datetime('2016-02-01 00:00:00') + pd.Timedelta(i, 'h') for i in range(24*335)]
test_timestamp_lst = [pd.to_datetime('2017-01-01 00:00:00') + pd.Timedelta(i, 'h') for i in range(24*181)]

pd_train_time = pd_time_information(train_timestamp_lst)
train_timestamp = pd_train_time.index

pd_test_time = pd_time_information(test_timestamp_lst)
test_timestamp = pd_test_time.index

def temporal_process(pd_time):
    np_hour = pd_time.loc[:, 'hour'].values
    hour = []
    for j in range(np_hour.shape[0]):
        hour.append(np.sin(2 * np.pi * np_hour[j] / 23.0))
    hour = np.array(hour).reshape((np.array(hour).shape[0], 1))

    np_day_of_week = pd_time.loc[:, 'day_of_week'].values
    day_of_week = []
    for j in range(np_day_of_week.shape[0]):
        day_of_week.append(np.sin(2 * np.pi * np_day_of_week[j] / 7.0))
    day_of_week = np.array(day_of_week).reshape((np.array(day_of_week).shape[0], 1))

    hour = np.tile(hour, (54, 1))
    day_of_week = np.tile(day_of_week, (54, 1))

    return hour, day_of_week

np_hour, np_day_of_week = temporal_process(pd_train_time)
np_hour_test, np_day_of_week_test = temporal_process(pd_test_time)

np_train = np.hstack((np_elec, np_hour, np_day_of_week))
np_test = np.hstack((np_elec_test, np_hour_test, np_day_of_week_test))

train_block = 24 * 335
test_block = 24 * 181
num_buildings = 54

for b in range(num_buildings):
    train_start = b * train_block
    train_end = (b+1) * train_block

    test_start = b * test_block
    test_end = (b+1) * test_block

    for i in range(np_train.shape[1]):
        min_data = np.min(np_train[train_start:train_end, i])
        max_data = np.max(np_train[train_start:train_end, i])

        if i == 1:
            np.save(f'data/min_max/min_{b}.npy', min_data)
            np.save(f'data/min_max/max_{b}.npy', max_data)

        np_train[train_start:train_end, i] = (np_train[train_start:train_end, i] - min_data) / (max_data - min_data)
        np_test[test_start:test_end, i] = (np_test[test_start:test_end, i] - min_data) / (max_data - min_data)

data_train = np.zeros((334*54, 48, 4))
data_test = np.zeros((180*54, 48, 4))

for N in range(num_buildings):
    for j in range((24*335-48)//24+1):
        idx = j
        for k in range(4):
            data_train[idx + N*334, :, k] = np_train[(j*24 + 24*335*N):(j*24 + 48 + 24*335*N), k]

    for l in range((24*181-48)//24+1):
        idx = l
        for m in range(4):
            data_test[idx + N*180, :, m] = np_test[(l*24 + 24*181*N):(l*24 + 48 + 24*181*N), m]

data_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))
data_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

B, D, T_win, F = data_train.shape
B_test, D_test, T_win_test, F_test = data_test.shape

building_idx = np.repeat(np.arange(B//334), 334)
building_idx_test = np.repeat(np.arange(B_test//180), 180)

np.save(f'data/train_data/mixed_train.npy', data_train)
np.save(f'data/train_data/mixed_test.npy', data_test)

np.save(f'data/train_data/mixed_building_idx_train.npy', building_idx)
np.save(f'data/train_data/mixed_building_idx_test.npy', building_idx_test)
