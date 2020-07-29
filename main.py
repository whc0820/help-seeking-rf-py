import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

PATH_SRC = '17_to_19_standardization_v3_cluster.csv'
COLNAMES_X = ['TM', 'TMA', 'AverageTip', 'RST', 'RSI',
              'RSE', 'Time', 'CorrectSteps', 'TriedSteps']
COLNAME_Y = 'Score'
N_SPLITS = 5

df_src = pd.read_csv(PATH_SRC)

kf = KFold(n_splits=N_SPLITS)


def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_kf_split_data(n, c):
    cluster = df_src.loc[df_src[f'Cluster{n}'] == c]
    cluster_x = cluster.loc[:, COLNAMES_X].to_numpy()
    cluster_y = cluster[COLNAME_Y].to_numpy()
    kf_cluster = []

    for train_index, test_index in kf.split(cluster_x):
        x_train = cluster_x[train_index]
        x_test = cluster_x[test_index]
        y_train = cluster_y[train_index]
        y_test = cluster_y[test_index]

        kf_cluster.append({
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test
        })

    return kf_cluster


# k = 3
n3_c1 = get_kf_split_data(3, 1)
n3_c2 = get_kf_split_data(3, 2)
n3_c3 = get_kf_split_data(3, 3)

n3_rmses = []
n3_c1_rmses = []
n3_c2_rmses = []
n3_c3_rmses = []

# k = 5
n5_c1 = get_kf_split_data(5, 1)
n5_c2 = get_kf_split_data(5, 2)
n5_c3 = get_kf_split_data(5, 3)
n5_c4 = get_kf_split_data(5, 4)
n5_c5 = get_kf_split_data(5, 5)

n5_rmses = []
n5_c1_rmses = []
n5_c2_rmses = []
n5_c3_rmses = []
n5_c4_rmses = []
n5_c5_rmses = []

for i in range(0, N_SPLITS):
    print(f'########## KFold {i} ##########\n')

    # k = 3, cluster1
    n3_c1_x_train = n3_c1[i]['x_train']
    n3_c1_x_test = n3_c1[i]['x_test']
    n3_c1_y_train = n3_c1[i]['y_train']
    n3_c1_y_test = n3_c1[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n3_c1_x_train, n3_c1_y_train)
    n3_c1_y_pred = regr.predict(n3_c1_x_test)
    n3_c1_rmse = mean_squared_error(n3_c1_y_test, n3_c1_y_pred, squared=False)
    n3_c1_rmses.append(n3_c1_rmse)
    print(f'n=3, c=1, {n3_c1_rmse}\n')

    # k = 3, cluster2
    n3_c2_x_train = n3_c2[i]['x_train']
    n3_c2_x_test = n3_c2[i]['x_test']
    n3_c2_y_train = n3_c2[i]['y_train']
    n3_c2_y_test = n3_c2[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n3_c2_x_train, n3_c2_y_train)
    n3_c2_y_pred = regr.predict(n3_c2_x_test)
    n3_c2_rmse = mean_squared_error(n3_c2_y_test, n3_c2_y_pred, squared=False)
    n3_c2_rmses.append(n3_c2_rmse)
    print(f'n=3, c=2, {n3_c2_rmse}\n')

    # k = 3, cluster3
    n3_c3_x_train = n3_c3[i]['x_train']
    n3_c3_x_test = n3_c3[i]['x_test']
    n3_c3_y_train = n3_c3[i]['y_train']
    n3_c3_y_test = n3_c3[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n3_c3_x_train, n3_c3_y_train)
    n3_c3_y_pred = regr.predict(n3_c3_x_test)
    n3_c3_rmse = mean_squared_error(n3_c3_y_test, n3_c3_y_pred, squared=False)
    n3_c3_rmses.append(n3_c3_rmse)
    print(f'n=3, c=3, {n3_c3_rmse}\n')

    # k = 3, all cluster
    n3_x_train = np.concatenate((n3_c1_x_train, n3_c2_x_train, n3_c3_x_train))
    n3_x_test = np.concatenate((n3_c1_x_test, n3_c2_x_test, n3_c3_x_test))
    n3_y_train = np.concatenate((n3_c1_y_train, n3_c2_y_train, n3_c3_y_train))
    n3_y_test = np.concatenate((n3_c1_y_test, n3_c2_y_test, n3_c3_y_test))

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n3_x_train, n3_y_train)
    n3_y_pred = regr.predict(n3_x_test)
    n3_rmse = mean_squared_error(n3_y_test, n3_y_pred, squared=False)
    n3_rmses.append(n3_rmse)
    print(f'n=3, c=all, {n3_rmse}\n')

    # k = 5, cluster1
    n5_c1_x_train = n5_c1[i]['x_train']
    n5_c1_x_test = n5_c1[i]['x_test']
    n5_c1_y_train = n5_c1[i]['y_train']
    n5_c1_y_test = n5_c1[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n5_c1_x_train, n5_c1_y_train)
    n5_c1_y_pred = regr.predict(n5_c1_x_test)
    n5_c1_rmse = mean_squared_error(n5_c1_y_test, n5_c1_y_pred, squared=False)
    n5_c1_rmses.append(n5_c1_rmse)
    print(f'n=5, c=1, {n5_c1_rmse}\n')

    # k = 5, cluster2
    n5_c2_x_train = n5_c2[i]['x_train']
    n5_c2_x_test = n5_c2[i]['x_test']
    n5_c2_y_train = n5_c2[i]['y_train']
    n5_c2_y_test = n5_c2[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n5_c2_x_train, n5_c2_y_train)
    n5_c2_y_pred = regr.predict(n5_c2_x_test)
    n5_c2_rmse = mean_squared_error(n5_c2_y_test, n5_c2_y_pred, squared=False)
    n5_c2_rmses.append(n5_c2_rmse)
    print(f'n=5, c=2, {n5_c2_rmse}\n')

    # k = 5, cluster3
    n5_c3_x_train = n5_c3[i]['x_train']
    n5_c3_x_test = n5_c3[i]['x_test']
    n5_c3_y_train = n5_c3[i]['y_train']
    n5_c3_y_test = n5_c3[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n5_c3_x_train, n5_c3_y_train)
    n5_c3_y_pred = regr.predict(n5_c3_x_test)
    n5_c3_rmse = mean_squared_error(n5_c3_y_test, n5_c3_y_pred, squared=False)
    n5_c3_rmses.append(n5_c3_rmse)
    print(f'n=5, c=3, {n5_c3_rmse}\n')

    # k = 5, cluster4
    n5_c4_x_train = n5_c4[i]['x_train']
    n5_c4_x_test = n5_c4[i]['x_test']
    n5_c4_y_train = n5_c4[i]['y_train']
    n5_c4_y_test = n5_c4[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n5_c4_x_train, n5_c4_y_train)
    n5_c4_y_pred = regr.predict(n5_c4_x_test)
    n5_c4_rmse = mean_squared_error(n5_c4_y_test, n5_c4_y_pred, squared=False)
    n5_c4_rmses.append(n5_c4_rmse)
    print(f'n=5, c=4, {n5_c4_rmse}\n')

    # k = 5, cluster5
    n5_c5_x_train = n5_c5[i]['x_train']
    n5_c5_x_test = n5_c5[i]['x_test']
    n5_c5_y_train = n5_c5[i]['y_train']
    n5_c5_y_test = n5_c5[i]['y_test']

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n5_c5_x_train, n5_c5_y_train)
    n5_c5_y_pred = regr.predict(n5_c5_x_test)
    n5_c5_rmse = mean_squared_error(n5_c5_y_test, n5_c5_y_pred, squared=False)
    n5_c5_rmses.append(n5_c5_rmse)
    print(f'n=5, c=5, {n5_c5_rmse}\n')

    # k = 5, all cluster
    n5_x_train = np.concatenate((n5_c1_x_train, n5_c2_x_train, n5_c3_x_train, n5_c4_x_train, n5_c5_x_train))
    n5_x_test = np.concatenate((n5_c1_x_test, n5_c2_x_test, n5_c3_x_test, n5_c4_x_test, n5_c5_x_test))
    n5_y_train = np.concatenate((n5_c1_y_train, n5_c2_y_train, n5_c3_y_train, n5_c4_y_train, n5_c5_y_train))
    n5_y_test = np.concatenate((n5_c1_y_test, n5_c2_y_test, n5_c3_y_test, n5_c4_y_test, n5_c5_y_test))

    regr = RandomForestRegressor(random_state=0)
    regr.fit(n5_x_train, n5_y_train)
    n5_y_pred = regr.predict(n5_x_test)
    n5_rmse = mean_squared_error(n5_y_test, n5_y_pred, squared=False)
    n5_rmses.append(n5_rmse)
    print(f'n=5, c=all, {n5_rmse}\n')


print(f'########## mean ##########\n')

print(f'n=3, c=1, {np.mean(np.array(n3_c1_rmses))}\n')
print(f'n=3, c=2, {np.mean(np.array(n3_c2_rmses))}\n')
print(f'n=3, c=3, {np.mean(np.array(n3_c3_rmses))}\n')
print(f'n=3, c=all, {np.mean(np.array(n3_rmses))}\n')

print(f'n=5, c=1, {np.mean(np.array(n5_c1_rmses))}\n')
print(f'n=5, c=2, {np.mean(np.array(n5_c2_rmses))}\n')
print(f'n=5, c=3, {np.mean(np.array(n5_c3_rmses))}\n')
print(f'n=5, c=4, {np.mean(np.array(n5_c4_rmses))}\n')
print(f'n=5, c=5, {np.mean(np.array(n5_c5_rmses))}\n')
print(f'n=5, c=all, {np.mean(np.array(n5_rmses))}\n')
