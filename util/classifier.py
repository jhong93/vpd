import copy
from abc import ABC
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.nn.utils.rnn import (
    pad_packed_sequence, pack_sequence, pack_padded_sequence)
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from .torch.cyclic_scheduler import CyclicLRWithRestarts
from .torch.batchnorm1d import MaskedBatchNorm1d


def pad_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = pad_packed_sequence(
        pack_sequence(data, enforce_sorted=False), batch_first=True)
    targets = torch.LongTensor([item[1] for item in batch])
    return data, targets


class _ABCSeqModel(ABC):

    class Seq(nn.Module):

        def __init__(self, cell_type, emb_dim, hidden_dim, num_classes,
                     depth=2, dropout=0.5, input_dropout=0.2,
                     input_batchnorm=False, use_attention=True):
            super().__init__()
            print('Backbone:', cell_type)
            print('  input dim:', emb_dim)
            print('  hidden dim:', hidden_dim)
            print('  out dim:', num_classes)
            print('  dropout:', dropout)
            print('  input dropout:', input_dropout)
            print('  attention:', use_attention)

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

            self.drop_in = nn.Dropout(p=input_dropout)
            self.bn_in = MaskedBatchNorm1d(emb_dim) if input_batchnorm else None

            hidden2 = 2 * hidden_dim
            self.fc_out = nn.Sequential(
                nn.BatchNorm1d(hidden2),
                nn.Dropout(p=dropout),
                nn.Linear(hidden2, hidden2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden2),
                nn.Dropout(p=dropout),
                nn.Linear(hidden2, num_classes))

            if use_attention:
                self.fc_attn = nn.Sequential(
                    nn.Linear(2 * depth * hidden_dim, hidden2),
                    nn.ReLU())

        def forward(self, x, xl):
            x = self.drop_in(x)
            if self.bn_in is not None:
                x = self.bn_in(x.permute(0, 2, 1), xl).permute(0, 2, 1)

            x = pack_padded_sequence(
                x, xl.cpu(), batch_first=True, enforce_sorted=False)

            output, last_state = self.backbone(x)
            use_attn = hasattr(self, 'fc_attn')
            output, _ = pad_packed_sequence(
                output, batch_first=True,
                padding_value=0 if use_attn else float('-inf'))

            if use_attn:
                if self.cell_type == 'lstm':
                    last_state = last_state[0]  # lstm has hidden state and cell
                n, _, h = last_state.shape
                last_state = torch.reshape(
                    last_state.permute(1, 0, 2), (-1, n * h))
                attn = self.fc_attn(last_state)
                attn = torch.bmm(output, attn.unsqueeze(2))
                attn = F.softmax(attn, 1)
                x = torch.bmm(output.permute(0, 2, 1), attn).squeeze(2)
            else:
                x = F.max_pool1d(
                    output.permute(0, 2, 1), output.shape[1]).squeeze(2)
            return self.fc_out(x)

    class CNN(nn.Module):

        def __init__(self, emb_dim, hidden_dim, num_classes,
                        kernel_sizes=[3, 5, 7], depth=1, dropout=0.5,
                        input_dropout=0.2):
            super().__init__()
            assert depth <= 2

            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, hidden_dim, k) for k in kernel_sizes])
            if depth > 1:
                self.convs2 = nn.ModuleList([
                    nn.Conv1d(hidden_dim, hidden_dim, 7, stride=k // 2)
                    for k in kernel_sizes])

            self.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(len(kernel_sizes) * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes))
            self.drop_in = nn.Dropout(p=input_dropout)

        def forward(self, x, _):
            x = self.drop_in(x)
            x = x.permute(0, 2, 1)
            x = [F.relu(conv(x)) for conv in self.convs]
            if hasattr(self, 'convs2'):
                x = [F.relu(conv(x)) for conv, x in zip(self.convs2, x)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
            x = torch.cat(x, 1)
            return self.fc(x)

    class Dataset(Dataset):

        def __init__(self, X, y):
            self.X = [torch.FloatTensor(x) for x in X]
            self.y = torch.LongTensor(y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

        def __len__(self):
            return len(self.X)

    @staticmethod
    def make_model(X, y, arch_type, hidden_dim, **kwargs):
        num_classes = np.unique(y).shape[0]
        emb_dim = X[0].shape[-1]
        if arch_type == 'cnn':
            model = _ABCSeqModel.CNN(
                emb_dim, hidden_dim, num_classes, **kwargs)
        else:
            model = _ABCSeqModel.Seq(
                arch_type, emb_dim, hidden_dim, num_classes, **kwargs)
        return model

    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        model.eval()

    def predict(self, x, full=False):
        x = torch.unsqueeze(torch.Tensor(x).to(self.device), 0)
        with torch.no_grad():
            pred = F.softmax(self.model(
                x, torch.LongTensor([x.shape[1]])
            ), dim=1).squeeze()
            if full:
                return pred.cpu().numpy()
            pred_cls = torch.argmax(pred).item()
            return pred_cls, pred[pred_cls].item()

    def predict_n(self, *xs):
        all_pred = []
        for x in xs:
            all_pred.append(self.predict(x, full=True))
        scores = np.mean(all_pred, axis=0)
        pred_cls = np.argmax(scores)
        return pred_cls, scores[pred_cls]


class BaseSeqModel(_ABCSeqModel):

    def __init__(self, arch_type, X, y, hidden_dim,
                 batch_size=50, num_epochs=500, min_epochs=10, wr_count=10,
                 early_term_acc=1, X_val=None, y_val=None, val_freq=1,
                 early_term_val_num_epochs=200, learning_rate=0.001,
                 load_weights=None, **kwargs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = _ABCSeqModel.make_model(X, y, arch_type, hidden_dim, **kwargs)
        print('Num parameters:', sum(p.numel() for p in model.parameters()))

        if load_weights:
            print('Loading:', load_weights)
            model.to(device)
            model.load_state_dict(torch.load(load_weights, map_location=device))
            best_model = model
        else:
            scaler = amp.GradScaler() if device == 'cuda' else None
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = CyclicLRWithRestarts(
                optimizer, batch_size, len(X),
                restart_period=num_epochs // wr_count)

            train_loader = DataLoader(
                _ABCSeqModel.Dataset(X, y), batch_size=batch_size,
                collate_fn=pad_collate, shuffle=True)

            # Model selection
            best_model = None
            best_val_err_loss = (1, float('inf'))
            best_val_epoch = 0
            if X_val is not None:
                val_loader = DataLoader(
                    _ABCSeqModel.Dataset(X_val, y_val), batch_size=batch_size,
                    collate_fn=pad_collate, shuffle=False)

            model.to(device)
            with trange(num_epochs) as pbar:
                def refresh_loss(l, a, cva=None, bva=None, bva_epoch=None):
                    pbar.set_description(
                        'Train {} (tl={:0.3f}, ta={:0.1f}{})'.format(
                            arch_type.upper(), l, a * 100,
                            '' if cva is None else ', va={:0.1f}, bva={:0.1f} @{}'.format(
                                cva * 100, bva * 100, bva_epoch)))
                    pbar.refresh()

                for epoch in pbar:
                    loss, acc = BaseSeqModel._epoch(
                        model, train_loader, device, optimizer, scaler, scheduler)
                    if X_val is not None:
                        if epoch % val_freq == 0:
                            val_loss, val_acc = BaseSeqModel._epoch(
                                model, val_loader, device)
                            if (1 - val_acc, val_loss) <= best_val_err_loss:
                                best_val_epoch = epoch
                                best_val_err_loss = (1 - val_acc, val_loss)
                                best_model = copy.deepcopy(model).to('cpu')
                            elif (
                                    early_term_val_num_epochs > 0 and
                                    epoch - early_term_val_num_epochs > best_val_epoch
                            ):
                                break
                        refresh_loss(loss, acc, val_acc,
                                    1 - best_val_err_loss[0], best_val_epoch)

                    else:
                        refresh_loss(loss, acc)

                    if epoch >= min_epochs and acc > early_term_acc:
                        break

        # Set trained model
        super().__init__(model if best_model is None else best_model, device)

    @staticmethod
    def _epoch(model, data_loader, device,
               optimizer=None, scaler=None, scheduler=None):
        model.eval() if optimizer is None else model.train()

        epoch_loss = 0.
        n_correct = 0
        with torch.no_grad() if optimizer is None else nullcontext():
            if scheduler is not None:
                scheduler.step()

            n = 0
            for X_pack, y in data_loader:
                X, X_len = X_pack

                with nullcontext() if scaler is None else amp.autocast():
                    pred = model(X.to(device), X_len.to(device))
                    y = y.to(device)
                    loss = F.cross_entropy(pred, y)

                if optimizer is not None:
                    if scaler is None:
                        loss.backward()
                        optimizer.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    if scheduler is not None:
                        scheduler.batch_step()
                    optimizer.zero_grad()

                epoch_loss += loss.cpu().item()
                n_correct += torch.sum(torch.argmax(pred, 1) == y).cpu().item()
                n += X.shape[0]
        return epoch_loss / n, n_correct / n

    def save(self, out_path):
        torch.save(self.model.state_dict(), out_path)
