import os.path

import numpy as np
import pandas as pd
import pm4py
import torch
from karateclub import Graph2Vec, FGSD, LDP
import xgboost as xgb

from ocpa.objects.log.importer.csv import factory as csv_import_factory
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP

from model import GAT, CustomPipeline, train_loop, evaluate, GCN
from preprocessing import get_process_executions_nx, generate_matrix_dataset, get_subgraphs_labeled

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main_function(filename, parameters, subg_funcs, g_funcs, no_feat, ks, embedding_name, embedding_size,
                  batch_size, learning_rate, mlp_hidden_dim, mlp_num_layers):
    # OCEL file import
    print(f"Importing OCEL {filename}")
    file_path = os.path.join('.', 'ocel', 'csv', filename)
    ocel = csv_import_factory.apply(file_path=file_path + '.csv', parameters=parameters)

    # Feature and target definition
    print("Preprocessing")
    activities = list(set(ocel.log.log["event_activity"].tolist()))

    # Event level targets
    event_target_set = [(predictive_monitoring.EVENT_REMAINING_TIME, ())]

    # Execution level targets (Outcome related)
    execution_target_set = [(predictive_monitoring.EXECUTION_NUM_OBJECT, ()),
                            (predictive_monitoring.EXECUTION_NUM_OF_END_EVENTS, ()),
                            (predictive_monitoring.EXECUTION_NUM_OF_EVENTS, ()),
                            (predictive_monitoring.EXECUTION_UNIQUE_ACTIVITIES, ())]

    if no_feat:
        execution_feature_set = []
        event_feature_set = []
    else:
        # Event level features
        event_feature_set = [(predictive_monitoring.EVENT_ELAPSED_TIME, ()),
                             (predictive_monitoring.EVENT_NUM_OF_OBJECTS, ())] + \
                            [(predictive_monitoring.EVENT_ACTIVITY, (act,)) for act in activities]
        # [(predictive_monitoring.EVENT_PRECEDING_ACTIVITIES, (act,)) for act in activities] + \
        execution_feature_set = []
    PE_nx = get_process_executions_nx(ocel, event_feature_set, event_target_set, execution_feature_set,
                                      execution_target_set)

    for k in ks:
        # Create result dataframe
        task_names = [subg_funcs[0].__name__] + [f.__name__ for f in g_funcs]

        print(f"Using subgraphs of length {k}")
        subgraph_list = get_subgraphs_labeled(PE_nx, k=k, subg_funcs=subg_funcs,
                                              g_funcs=g_funcs)

        get_num_classes = lambda feature_idx: [len(np.unique([g[feature_idx] for g in subgraph_list]))]
        if len(subg_funcs) > 0:
            pred_types = [None]
        else:
            pred_types = []

        for i in range(len(g_funcs)):
            num_classes = get_num_classes(2 + len(subg_funcs) + i)
            if num_classes[0] == 1:
                task_names.pop(len(subg_funcs) + i)
            if num_classes[0] > 1:
                pred_types += num_classes

        result_dfs = []
        result_dir_paths = []
        for task_name in task_names:
            result_df = pd.DataFrame(columns=['prediction_layer', 'score'])
            result_dfs.append(result_df)
            result_dir_path = os.path.join('.', 'results', filename, task_name, str(k))
            result_dir_paths.append(result_dir_path)
            os.makedirs(result_dir_path, exist_ok=True)

        y = np.array([g[2:] for g in subgraph_list])
        for i, pred_type in enumerate(pred_types):
            if pred_type is None:
                scaler = MinMaxScaler()
                y[:, i] = scaler.fit_transform(y[:, i].reshape(-1, 1))[:, 0]
                for idx, subgraph in enumerate(subgraph_list):
                    subgraph[2 + i] = float(y[idx, i])
            else:
                encoder = LabelEncoder()
                y[:, i] = encoder.fit_transform(y[:, i])
                for idx, subgraph in enumerate(subgraph_list):
                    subgraph[2 + i] = int(y[idx, i])

        input_dataset = generate_matrix_dataset(subgraph_list)
        num_features = input_dataset[0].x.shape[1]
        # train val test split
        temp_data, test_data = train_test_split(input_dataset, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(temp_data, test_size=0.2, random_state=42)
        # data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        # embedding, predictor setup
        print("Model Construction")
        # Init Embedding model
        if embedding_name == 'GAT':
            embedding_layer = GAT(num_layers=2, num_features=num_features, hidden_dim=16, target_size=embedding_size,
                                  heads=4)
        elif embedding_name == 'GCN':
            embedding_layer = GCN(num_layers=2, num_features=num_features, hidden_dim=16, target_size=embedding_size)
        elif embedding_name == 'Graph2Vec':
            embedding_layer = Graph2Vec(dimensions=embedding_size)
            embedding_layer.fit([t.graph for t in train_data])
        elif embedding_name == 'FGSD':
            embedding_layer = FGSD(hist_bins=embedding_size)
            embedding_layer.fit([t.graph for t in train_data])

        # Init predictors
        nn_preds = []
        for pred_type in pred_types:
            if pred_type is None:
                target_dim = 1
                mlp = MLP(in_channels=embedding_size, hidden_channels=mlp_hidden_dim, out_channels=target_dim,
                          num_layers=mlp_num_layers)
            else:
                target_dim = pred_type
                mlp = MLP(in_channels=embedding_size, hidden_channels=mlp_hidden_dim, out_channels=target_dim,
                          num_layers=mlp_num_layers)
            nn_preds.append(mlp)

        ml_preds = []
        for pred_type in pred_types:
            if pred_type is None:
                target_dim = 1
                ml = LinearRegression()
            else:
                target_dim = pred_type
                ml = LogisticRegression(max_iter=1000)
            ml_preds.append(ml)
        # Init pipeline
        model = CustomPipeline(embedding_layer, nn_preds, ml_preds)
        # Init optimizer
        optimizer = optim.Adam(model.parameters(), learning_rate)
        # Move model and data to device (cuda if exists)
        # model = model.to(device)
        print("Training...")
        train_loop(model, optimizer, train_loader, val_loader, pred_types, num_epochs=10)
        # Tests on linear ml models
        test_nn_scores, test_ml_scores = evaluate(model, test_loader, pred_types, train_loader)
        # Store test scores for nn and ml
        for i, result_df in enumerate(result_dfs):
            if i == 0:
                a = scaler.inverse_transform(np.array(test_nn_scores[i]).reshape(-1, 1))[0][0]
                res_row = [f"MLP_{mlp_hidden_dim}_{mlp_num_layers}", a/86400]
                result_df.loc[len(result_df)] = res_row
                b = scaler.inverse_transform(test_ml_scores[i].reshape(-1, 1))[0][0]
                res_row = ["Linear models", b/86400]
                result_df.loc[len(result_df)] = res_row
            else:
                res_row = [f"MLP_{mlp_hidden_dim}_{mlp_num_layers}", test_nn_scores[i]]
                result_df.loc[len(result_df)] = res_row
                res_row = ["Linear models", test_ml_scores[i]]
                result_df.loc[len(result_df)] = res_row
        # Tests on Random forest models
        new_ml_preds = []
        for pred_type in pred_types:
            if pred_type is None:
                target_dim = 1
                new_ml = RandomForestRegressor()
            else:
                target_dim = pred_type
                new_ml = RandomForestClassifier()
            new_ml_preds.append(new_ml)
        model.fit_new_ml_predictors(train_loader, new_ml_preds)
        _, test_ml_scores_rf = evaluate(model, test_loader, pred_types, train_loader)
        # Store scores
        for i, result_df in enumerate(result_dfs):
            if i == 0:
                a = scaler.inverse_transform(np.array(test_ml_scores[i]).reshape(-1, 1))[0][0]
                res_row = ["Random forest models", a/86400]
                result_df.loc[len(result_df)] = res_row
            else:
                res_row = ["Random forest models", test_ml_scores[i]]
                result_df.loc[len(result_df)] = res_row
        # Tests on XGBoost models
        new_ml_preds = []
        for pred_type in pred_types:
            if pred_type is None:
                target_dim = 1
                new_ml = xgb.XGBRegressor()
            else:
                target_dim = pred_type
                new_ml = xgb.XGBClassifier()
            new_ml_preds.append(new_ml)
        model.fit_new_ml_predictors(train_loader, new_ml_preds)
        _, test_ml_scores_rf = evaluate(model, test_loader, pred_types, train_loader)
        # Store scores
        for i, result_df in enumerate(result_dfs):
            if i == 0:
                a = scaler.inverse_transform(np.array(test_ml_scores[i]).reshape(-1, 1))[0][0]
                res_row = ["XGB models", a/86400]
                result_df.loc[len(result_df)] = res_row
            else:
                res_row = ["XGB models", test_ml_scores[i]]
                result_df.loc[len(result_df)] = res_row

        for i, result_df in enumerate(result_dfs):
            result_file_path = os.path.join(result_dir_paths[i], f"{embedding_name}_{embedding_size}_{no_feat}.csv")
            result_df.to_csv(result_file_path, mode='a')
            print(result_df)
