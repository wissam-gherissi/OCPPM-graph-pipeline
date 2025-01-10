import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAT(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden_dim, target_size, heads=4):
        super(GAT, self).__init__()
        self.hidden_size = hidden_dim
        self.num_features = num_features
        self.target_size = target_size
        self.num_layers = num_layers
        self.num_heads = heads
        self.dropout = 0.5
        self.convs = torch.nn.ModuleList([GATConv(self.num_features, self.hidden_size, heads=self.num_heads)] +
                                         [GATConv(self.hidden_size * self.num_heads, self.hidden_size * self.num_heads,
                                                  heads=self.num_heads)
                                          for _ in range(self.num_layers - 2)] +
                                         [GATConv(self.hidden_size * self.num_heads, self.target_size,
                                                  heads=self.num_heads, concat=False)])

    def forward(self, x, edge_index, batch):
        # Apply multiple GATConv layers
        for conv in self.convs:
            x = F.gelu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch=batch)

        return x


class GCN(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden_dim, target_size):
        super(GCN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_features = num_features
        self.target_size = target_size
        self.num_layers = num_layers
        self.dropout = 0.5
        self.convs = torch.nn.ModuleList([GCNConv(self.num_features, self.hidden_size)] +
                                         [GCNConv(self.hidden_size, self.hidden_size)
                                          for _ in range(self.num_layers - 2)] +
                                         [GCNConv(self.hidden_size, self.target_size)])

    def forward(self, x, edge_index, batch):
        # Apply multiple GATConv layers
        for conv in self.convs:
            x = F.gelu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch=batch)
        return x


class CustomPipeline(torch.nn.Module):
    def __init__(self, embedding_model, mlp_preds, ml_preds):
        super(CustomPipeline, self).__init__()
        self.Embedding = embedding_model
        self.nn_predictors = torch.nn.ModuleList(mlp_preds)
        self.ml_predictors = ml_preds
        self.uses_gnn = isinstance(embedding_model, torch.nn.Module)

    def forward(self, batch):
        embeddings = self.compute_embeddings(batch)
        predictions = [mlp(embeddings) for mlp in self.nn_predictors]
        return predictions

    def compute_embeddings(self, data):
        x, edge_index, graph, batch = data.x, data.edge_index, data.graph, data.batch
        if self.uses_gnn:
            embeddings = self.Embedding(x, edge_index, batch)
            return embeddings.clone().detach().requires_grad_(True)
        else:
            if not isinstance(graph, list):
                graph = [graph]
            self.Embedding.fit(graph)
            embeddings = self.Embedding.get_embedding()
            return torch.tensor(embeddings, dtype=torch.float)


    def fit_ml_predictors(self, data):
        y = torch.tensor([d.y for d in data.dataset])
        d = DataLoader(data.dataset, batch_size=len(data.dataset), pin_memory=True)
        for batch in d:
            batch = batch.to(device)
            embeddings = self.compute_embeddings(batch)
        for idx, ml_predictor in enumerate(self.ml_predictors):
            ml_predictor.fit(embeddings.cpu().detach().numpy(), np.array(y[:, idx]))

    def fit_new_ml_predictors(self, data, new_ml_predictors):
        self.ml_predictors = new_ml_predictors
        self.fit_ml_predictors(data)

    def predict(self, data):
        embeddings = self.compute_embeddings(data)
        nn_predictions = [mlp(embeddings) for mlp in self.nn_predictors]
        ml_predictions = [torch.tensor(ml.predict(np.array(embeddings.cpu()))).unsqueeze(-1) for ml in self.ml_predictors]
        return nn_predictions, ml_predictions


def train_loop(model, optimizer, train_loader, val_loader, pred_types, num_epochs=1):
    model.train()
    loss_functions = []
    for pred_type in pred_types:
        if pred_type is None:
            loss_functions.append(lambda x,y: nn.HuberLoss()(x, y.unsqueeze(-1)))
        else:
            loss_functions.append(lambda x,y: nn.CrossEntropyLoss()(x, y.to(torch.long)))
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            ground_truth = torch.tensor(batch.y, device=device)
            losses = 0.0
            for i, (loss_fn, preds) in enumerate(zip(loss_functions, predictions)):
                losses += loss_fn(preds, ground_truth[:, i])
            losses.backward()
            optimizer.step()
        evaluate(model, val_loader, pred_types, train_loader)


def evaluate(model, data, pred_types, training_data, new_ml_predictors=None):
    model.eval()
    loss_functions = []
    score_functions = []
    ground_truth = torch.tensor([d.y for d in data.dataset], device=device)
    for pred_type in pred_types:
        if pred_type is None:
            loss_functions.append(lambda x,y: nn.HuberLoss()(x, y.unsqueeze(-1)))
            score_functions.append(nn.L1Loss())
        else:
            loss_functions.append(lambda x,y: nn.CrossEntropyLoss()(x, y.to(torch.long)))
            score_functions.append(lambda x, y: accuracy_score(np.argmax(nn.functional.sigmoid(x).cpu(), axis=1), y.cpu()))
    if new_ml_predictors is None:
        model.fit_ml_predictors(training_data)
    else:
        model.fit_new_ml_predictors(training_data, new_ml_predictors)
    with torch.no_grad():
        nn_predictions_list, ml_predictions_list = [[] for pred_type in pred_types], [[] for pred_type in pred_types]
        for batch in data:
            batch = batch.to(device)
            nn_predictions, ml_predictions = model.predict(batch)
            for i, pred_type in enumerate(pred_types):
                nn_predictions_list[i].append(nn_predictions[i])
                ml_predictions_list[i].append(ml_predictions[i])
        nn_predictions = [torch.vstack(e) for e in nn_predictions_list]
        ml_predictions = [torch.vstack(e) for e in ml_predictions_list]
        nn_scores, ml_scores, losses = [], [], []
        for i, (score_fn, loss_fn) in enumerate(zip(score_functions, loss_functions)):
            nn_score = score_fn(nn_predictions[i], ground_truth[:, i].unsqueeze(-1))
            nn_scores.append(float(nn_score))
            ml_score = score_fn(ml_predictions[i], ground_truth[:, i].cpu().unsqueeze(-1))
            ml_scores.append(ml_score)
            loss = loss_fn(nn_predictions[i], ground_truth[:, i])
            losses.append(loss)
            # print(f"Task {i}:\tLoss: {loss}\tScore (MLP predictor): {nn_score}\tScore (ML predictor): {ml_score}")
        return nn_scores, ml_scores
