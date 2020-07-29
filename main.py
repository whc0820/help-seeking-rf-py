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


def get_rf_rmse(x_train, x_test, y_train, y_test):
    regr = RandomForestRegressor(random_state=0)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    return mean_squared_error(y_test, y_pred, squared=False)


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
    # k = 3, cluster1
    n3_c1_rmses.append(get_rf_rmse(
        n3_c1[i]['x_train'], n3_c1[i]['x_test'], n3_c1[i]['y_train'], n3_c1[i]['y_test']))

    # k = 3, cluster2
    n3_c2_rmses.append(get_rf_rmse(
        n3_c2[i]['x_train'], n3_c2[i]['x_test'], n3_c2[i]['y_train'], n3_c2[i]['y_test']))

    # k = 3, cluster3
    n3_c3_rmses.append(get_rf_rmse(
        n3_c3[i]['x_train'], n3_c3[i]['x_test'], n3_c3[i]['y_train'], n3_c3[i]['y_test']))

    # k = 3, all cluster
    n3_x_train = np.concatenate(
        (n3_c1[i]['x_train'], n3_c2[i]['x_train'], n3_c3[i]['x_train']))
    n3_x_test = np.concatenate(
        (n3_c1[i]['x_test'], n3_c2[i]['x_test'], n3_c3[i]['x_test']))
    n3_y_train = np.concatenate(
        (n3_c1[i]['y_train'], n3_c2[i]['y_train'], n3_c3[i]['y_train']))
    n3_y_test = np.concatenate(
        (n3_c1[i]['y_test'], n3_c2[i]['y_test'], n3_c3[i]['y_test']))
    n3_rmses.append(get_rf_rmse(n3_x_train, n3_x_test, n3_y_train, n3_y_test))

    # k = 5, cluster1
    n5_c1_rmses.append(get_rf_rmse(
        n5_c1[i]['x_train'], n5_c1[i]['x_test'], n5_c1[i]['y_train'], n5_c1[i]['y_test']))

    # k = 5, cluster2
    n5_c2_rmses.append(get_rf_rmse(
        n5_c2[i]['x_train'], n5_c2[i]['x_test'], n5_c2[i]['y_train'], n5_c2[i]['y_test']))

    # k = 5, cluster3
    n5_c3_rmses.append(get_rf_rmse(
        n5_c3[i]['x_train'], n5_c3[i]['x_test'], n5_c3[i]['y_train'], n5_c3[i]['y_test']))

    # k = 5, cluster4
    n5_c4_rmses.append(get_rf_rmse(
        n5_c4[i]['x_train'], n5_c4[i]['x_test'], n5_c4[i]['y_train'], n5_c4[i]['y_test']))

    # k = 5, cluster5
    n5_c5_rmses.append(get_rf_rmse(
        n5_c5[i]['x_train'], n5_c5[i]['x_test'], n5_c5[i]['y_train'], n5_c5[i]['y_test']))

    # k = 5, all cluster
    n5_x_train = np.concatenate(
        (n5_c1[i]['x_train'], n5_c2[i]['x_train'], n5_c3[i]['x_train'], n5_c4[i]['x_train'], n5_c5[i]['x_train']))
    n5_x_test = np.concatenate(
        (n5_c1[i]['x_test'], n5_c2[i]['x_test'], n5_c3[i]['x_test'], n5_c4[i]['x_test'], n5_c5[i]['x_test']))
    n5_y_train = np.concatenate(
        (n5_c1[i]['y_train'], n5_c2[i]['y_train'], n5_c3[i]['y_train'], n5_c4[i]['y_train'], n5_c5[i]['y_train']))
    n5_y_test = np.concatenate(
        (n5_c1[i]['y_test'], n5_c2[i]['y_test'], n5_c3[i]['y_test'], n5_c4[i]['y_test'], n5_c5[i]['y_test']))
    n5_rmses.append(get_rf_rmse(n5_x_train, n5_x_test, n5_y_train, n5_y_test))

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
