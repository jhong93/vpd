import random
import copy
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import trange


class BaseProposalModel:

    class _Seq(nn.Module):

        def __init__(self, cell_type, emb_dim, hidden_dim,
                     depth=2, dropout=0.5, input_dropout=0.2):
            super().__init__()

            print('Backbone:', cell_type)
            print('  input dim:', emb_dim)
            print('  hidden dim:', hidden_dim)
            print('  dropout:', dropout)
            print('  input dropout:', input_dropout)

            self.hidden_dim = hidden_dim
            self.cell_type = cell_type
            if cell_type == 'lstm':
                self.backbone = nn.LSTM(
                    emb_dim, hidden_dim, num_layers=depth, batch_first=True,
                    bidirectional=True)
            elif cell_type == 'gru':
                self.backbone = nn.GRU(
                    emb_dim, hidden_dim, num_layers=depth, batch_first=True,
                    bidirectional=True)
            else:
                raise NotImplementedError()

            self.fc_out = nn.Sequential(
                nn.BatchNorm1d(2 * hidden_dim),
                nn.Dropout(p=dropout),
                nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.Dropout(p=dropout),
                nn.Linear(2 * hidden_dim, 2))
            self.drop_in = nn.Dropout(p=input_dropout)

        def forward(self, x):
            x = self.drop_in(x)
            output, _ = self.backbone(x)
            return self.fc_out(torch.reshape(output, (-1, output.shape[-1])))

    class _Dataset(Dataset):

        def __init__(self, X, y, seq_len=250, n=5000):
            self.X = [torch.FloatTensor(xx) for xx in X]
            self.y = [torch.LongTensor(yy) for yy in y]
            self.weights = [max(0, len(z) - seq_len) for z in y]
            assert max(self.weights) > 0, 'All sequences are too short!'
            self.seq_len = seq_len
            self.n = n

        def __getitem__(self, unused):
            idx = random.choices(range(len(self.y)), weights=self.weights, k=1)[0]
            x = self.X[idx]
            y = self.y[idx]
            start_frame = random.randint(0, y.shape[0] - self.seq_len - 1)
            return (x[start_frame:start_frame + self.seq_len, :],
                    y[start_frame:start_frame + self.seq_len])

        def __len__(self):
            return self.n

    def __init__(self, arch_type, X, y, hidden_dim, batch_size=100,
                 num_epochs=25, min_epochs=10, early_term_acc=1,
                 early_term_no_val_improvement=50,
                 X_val=None, y_val=None, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        emb_dim = X[0].shape[-1]
        model = BaseProposalModel._Seq(arch_type, emb_dim, hidden_dim, **kwargs)
        optimizer = torch.optim.AdamW(model.parameters())

        train_loader = DataLoader(
            BaseProposalModel._Dataset(X, y), batch_size=batch_size)

        # Model selection
        best_model = None
        best_val_err_loss = (1, float('inf'))
        best_val_epoch = 0
        if X_val is not None:
            val_loader = DataLoader(
                BaseProposalModel._Dataset(X_val, y_val),
                batch_size=batch_size)

        model.to(self.device)
        with trange(num_epochs) as pbar:
            def refresh_loss(l, a, cva=None, bva=None, bva_epoch=None):
                pbar.set_description(
                    'Train {} (tl={:0.3f}, ta={:0.1f}{})'.format(
                        arch_type.upper(), l, a * 100, '' if cva is None else
                        ', va={:0.1f}, bva={:0.1f} @{}'.format(
                            cva * 100, bva * 100, bva_epoch)))
                pbar.refresh()

            for epoch in pbar:
                loss, acc = BaseProposalModel._epoch(
                    model, train_loader, self.device, optimizer)
                if X_val is not None:
                    val_loss, val_acc = BaseProposalModel._epoch(
                        model, val_loader, self.device)
                    if (1 - val_acc, val_loss) <= best_val_err_loss:
                        best_val_epoch = epoch
                        best_val_err_loss = (1 - val_acc, val_loss)
                        best_model = copy.deepcopy(model).to('cpu')
                        if (
                                1 - best_val_err_loss[0] >= early_term_acc
                                and epoch > min_epochs
                        ):
                            break
                    elif (
                            epoch - best_val_epoch >= early_term_no_val_improvement
                            and epoch > min_epochs
                    ):
                        break
                    refresh_loss(loss, acc, val_acc,
                                 1 - best_val_err_loss[0], best_val_epoch)

                else:
                    refresh_loss(loss, acc)

                if epoch >= min_epochs and acc > early_term_acc:
                    break

        if best_model is None:
            self.model = model
        else:
            self.model = best_model.to(self.device)
            del model
        self.model.eval()

    @staticmethod
    def _epoch(model, loader, device, optimizer=None):
        model.eval() if optimizer is None else model.train()

        epoch_loss = 0.
        n_correct = 0
        with torch.no_grad() if optimizer is None else nullcontext():
            n = 0
            nt = 0
            for X, y in loader:
                pred = model(X.to(device))
                y = y.flatten().to(device)
                loss = F.cross_entropy(pred, y)

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.cpu().item()
                n_correct += torch.sum(torch.argmax(pred, 1) == y).cpu().item()
                nt += y.shape[0]
                n += X.shape[0]
        return epoch_loss / n, n_correct / nt

    def predict(self, x):
        x = torch.unsqueeze(torch.Tensor(x).to(self.device), 0)
        with torch.no_grad():
            pred = F.softmax(self.model(x), dim=1).squeeze()
            return pred[:, 1].cpu().numpy()

    @staticmethod
    def get_proposals(scores, activation_thresh, min_prop_len=3,
                      merge_thresh=1):
        props = []
        curr_prop = None
        for i in range(len(scores)):
            if scores[i] >= activation_thresh:
                if curr_prop is None:
                    curr_prop = (i, i)
                else:
                    curr_prop = (curr_prop[0], i)
            else:
                if curr_prop is not None:
                    props.append(curr_prop)
                    curr_prop = None
        if curr_prop is not None:
            props.append(curr_prop)
            del curr_prop

        merged_props = []
        for p in props:
            if len(merged_props) == 0:
                merged_props.append(p)
            else:
                last_p = merged_props[-1]
                if p[0] - last_p[1] <= merge_thresh:
                    merged_props[-1] = (last_p[0], p[1])
                else:
                    merged_props.append(p)

        def get_score(a, b):
            return np.mean(scores[a:b + 1])

        return  [(p, get_score(*p)) for p in merged_props
                 if p[1] - p[0] > min_prop_len]


class EnsembleProposalModel:

    def __init__(self, arch_type, X, y, hidden_dim, ensemble_size=3, splits=5,
                 custom_split=None, **kwargs):
        if ensemble_size > 1:
            print('Training an ensemble of {} {}s with {} folds'.format(
                ensemble_size, arch_type.upper(), splits))
        else:
            print('Holding out 1 / {} for validation'.format(splits))
        kf = KFold(n_splits=splits, shuffle=True)

        if custom_split is None:
            custom_split = np.arange(len(X))
        unique_idxs = list(set(custom_split))

        models = []
        for train, val in kf.split(unique_idxs):
            train = set(train)
            val = set(val)
            X_train, y_train = zip(*[(X[j], y[j]) for j in range(len(X))
                                      if custom_split[j] in train])
            X_val, y_val = zip(*[(X[j], y[j]) for j in range(len(X))
                                  if custom_split[j] in val])
            models.append(BaseProposalModel(
                arch_type, X_train, y_train, hidden_dim,
                X_val=X_val, y_val=y_val, **kwargs))
            if len(models) >= ensemble_size:
                break
        self.models = models

    def predict(self, x):
        return self.predict_n(x)

    def predict_n(self, *xs):
        pred = None
        denom = 0
        for model in self.models:
            for x in xs:
                tmp = model.predict(x)
                if pred is None:
                    pred = tmp
                else:
                    pred += tmp
                denom += 1
        return pred / denom