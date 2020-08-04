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

n3_c1_rmses = []
n3_c2_rmses = []
n3_c3_rmses = []

n3_c1_g_rmses = []
n3_c2_g_rmses = []
n3_c3_g_rmses = []


# k = 5
n5_c1 = get_kf_split_data(5, 1)
n5_c2 = get_kf_split_data(5, 2)
n5_c3 = get_kf_split_data(5, 3)
n5_c4 = get_kf_split_data(5, 4)
n5_c5 = get_kf_split_data(5, 5)

n5_c1_rmses = []
n5_c2_rmses = []
n5_c3_rmses = []
n5_c4_rmses = []
n5_c5_rmses = []

n5_c1_g_rmses = []
n5_c2_g_rmses = []
n5_c3_g_rmses = []
n5_c4_g_rmses = []
n5_c5_g_rmses = []


for i in range(0, N_SPLITS):
    # k = 3, general, train data
    n3_g_x_train = np.concatenate(
        (n3_c1[i]['x_train'], n3_c2[i]['x_train'], n3_c3[i]['x_train']))
    n3_g_y_train = np.concatenate(
        (n3_c1[i]['y_train'], n3_c2[i]['y_train'], n3_c3[i]['y_train']))

    # k = 3, cluster1
    n3_c1_rmse = get_rf_rmse(
        n3_c1[i]['x_train'], n3_c1[i]['x_test'], n3_c1[i]['y_train'], n3_c1[i]['y_test'])
    n3_c1_rmses.append(n3_c1_rmse)

    n3_c1_g_rmse = get_rf_rmse(
        n3_g_x_train, n3_c1[i]['x_test'], n3_g_y_train, n3_c1[i]['y_test'])
    n3_c1_g_rmses.append(n3_c1_g_rmse)

    # k = 3, cluster2
    n3_c2_rmse = get_rf_rmse(
        n3_c2[i]['x_train'], n3_c2[i]['x_test'], n3_c2[i]['y_train'], n3_c2[i]['y_test'])
    n3_c2_rmses.append(n3_c2_rmse)

    n3_c2_g_rmse = get_rf_rmse(
        n3_g_x_train, n3_c2[i]['x_test'], n3_g_y_train, n3_c2[i]['y_test'])
    n3_c2_g_rmses.append(n3_c2_g_rmse)

    # k = 3, cluster3
    n3_c3_rmse = get_rf_rmse(
        n3_c3[i]['x_train'], n3_c3[i]['x_test'], n3_c3[i]['y_train'], n3_c3[i]['y_test'])
    n3_c3_rmses.append(n3_c3_rmse)

    n3_c3_g_rmse = get_rf_rmse(
        n3_g_x_train, n3_c3[i]['x_test'], n3_g_y_train, n3_c3[i]['y_test'])
    n3_c3_g_rmses.append(n3_c3_g_rmse)

    # k = 5, general train data
    n5_g_x_train = np.concatenate(
        (n5_c1[i]['x_train'], n5_c2[i]['x_train'], n5_c3[i]['x_train'], n5_c4[i]['x_train'], n5_c5[i]['x_train']))
    n5_g_y_train = np.concatenate(
        (n5_c1[i]['y_train'], n5_c2[i]['y_train'], n5_c3[i]['y_train'], n5_c4[i]['y_train'], n5_c5[i]['y_train']))

    # k = 5, cluster1
    n5_c1_rmse = get_rf_rmse(
        n5_c1[i]['x_train'], n5_c1[i]['x_test'], n5_c1[i]['y_train'], n5_c1[i]['y_test'])
    n5_c1_rmses.append(n5_c1_rmse)

    n5_c1_g_rmse = get_rf_rmse(
        n5_g_x_train, n5_c1[i]['x_test'], n5_g_y_train, n5_c1[i]['y_test'])
    n5_c1_g_rmses.append(n5_c1_g_rmse)

    # k = 5, cluster2
    n5_c2_rmse = get_rf_rmse(
        n5_c2[i]['x_train'], n5_c2[i]['x_test'], n5_c2[i]['y_train'], n5_c2[i]['y_test'])
    n5_c2_rmses.append(n5_c2_rmse)

    n5_c2_g_rmse = get_rf_rmse(
        n5_g_x_train, n5_c2[i]['x_test'], n5_g_y_train, n5_c2[i]['y_test'])
    n5_c2_g_rmses.append(n5_c2_g_rmse)

    # k = 5, cluster3
    n5_c3_rmse = get_rf_rmse(
        n5_c3[i]['x_train'], n5_c3[i]['x_test'], n5_c3[i]['y_train'], n5_c3[i]['y_test'])
    n5_c3_rmses.append(n5_c3_rmse)

    n5_c3_g_rmse = get_rf_rmse(
        n5_g_x_train, n5_c3[i]['x_test'], n5_g_y_train, n5_c3[i]['y_test'])
    n5_c3_g_rmses.append(n5_c3_g_rmse)

    # k = 5, cluster4
    n5_c4_rmse = get_rf_rmse(
        n5_c4[i]['x_train'], n5_c4[i]['x_test'], n5_c4[i]['y_train'], n5_c4[i]['y_test'])
    n5_c4_rmses.append(n5_c4_rmse)

    n5_c4_g_rmse = get_rf_rmse(
        n5_g_x_train, n5_c4[i]['x_test'], n5_g_y_train, n5_c4[i]['y_test'])
    n5_c4_g_rmses.append(n5_c4_g_rmse)

    # k = 5, cluster5
    n5_c5_rmse = get_rf_rmse(
        n5_c5[i]['x_train'], n5_c5[i]['x_test'], n5_c5[i]['y_train'], n5_c5[i]['y_test'])
    n5_c5_rmses.append(n5_c5_rmse)

    n5_c5_g_rmse = get_rf_rmse(
        n5_g_x_train, n5_c5[i]['x_test'], n5_g_y_train, n5_c5[i]['y_test'])
    n5_c5_g_rmses.append(n5_c5_g_rmse)


out_df = pd.DataFrame(np.array([n3_c1_rmses, n3_c1_g_rmses,
                                n3_c2_rmses, n3_c2_g_rmses,
                                n3_c3_rmses, n3_c3_g_rmses,
                                n5_c1_rmses, n5_c1_g_rmses,
                                n5_c2_rmses, n5_c2_g_rmses,
                                n5_c3_rmses, n5_c3_g_rmses,
                                n5_c4_rmses, n5_c4_g_rmses,
                                n5_c5_rmses, n5_c5_g_rmses]),
                      columns=['k-fold-1', 'k-fold-2', 'k-fold-3', 'k-fold-4', 'k-fold-5'])
out_df.insert(0, 'k-means-n', ['3', '3', '3', '3', '3',
                               '3', '5', '5', '5', '5', '5', '5', '5', '5', '5', '5'])
out_df.insert(1, 'model', ['cluster1', 'cluster1-general', 'cluster2', 'cluster2-general', 'cluster3', 'cluster3-general',
                           'cluster1', 'cluster1-general', 'cluster2', 'cluster2-general', 'cluster3', 'cluster3-general', 'cluster4', 'cluster4-general', 'cluster5', 'cluster5-general'])
out_df['mean'] = [np.mean(np.array(n3_c1_rmses)),
                  np.mean(np.array(n3_c1_g_rmses)),
                  np.mean(np.array(n3_c2_rmses)),
                  np.mean(np.array(n3_c2_g_rmses)),
                  np.mean(np.array(n3_c3_rmses)),
                  np.mean(np.array(n3_c3_g_rmses)),
                  np.mean(np.array(n5_c1_rmses)),
                  np.mean(np.array(n5_c1_g_rmses)),
                  np.mean(np.array(n5_c2_rmses)),
                  np.mean(np.array(n5_c2_g_rmses)),
                  np.mean(np.array(n5_c3_rmses)),
                  np.mean(np.array(n5_c3_g_rmses)),
                  np.mean(np.array(n5_c4_rmses)),
                  np.mean(np.array(n5_c4_g_rmses)),
                  np.mean(np.array(n5_c5_rmses)),
                  np.mean(np.array(n5_c5_g_rmses))]
out_df.to_csv(path_or_buf='out.csv', index=False)