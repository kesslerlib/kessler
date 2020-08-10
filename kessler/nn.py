import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from . import util


class DatasetEventSet(Dataset):
    def __init__(self, event_set, features, features_stats=None):
        self._event_set = event_set
        self._max_event_length = max(map(len, self._event_set))
        self._features = features
        self._features_length = len(features)
        if features_stats is None:
            df = event_set.to_dataframe()
            features_numpy = df[features].to_numpy()
            mean = features_numpy.mean(0)
            stddev = features_numpy.std(0)
            self._features_stats = {'mean': mean, 'stddev': stddev}
        else:
            self._features_stats = features_stats

    def __len__(self):
        return len(self._event_set)

    # single item (from Dataset): max_event_length x features
    # minibatch of several items (from DataLoader): batch_size x max_event_length x features
    def __getitem__(self, i):
        event = self._event_set[i]
        x = torch.zeros(self._max_event_length, self._features_length)
        for i, cdm in enumerate(event):
            x[i] = torch.tensor([(cdm[feature]-self._features_stats['mean'][j])/(self._features_stats['stddev'][j]+1e-8) for j, feature in enumerate(self._features)])
        return x, torch.tensor(len(event))


class LSTMPredictor(nn.Module):
    def __init__(self, lstm_size=256, lstm_depth=2, dropout=None, features=None):
        super().__init__()

        self.input_size = len(features)
        self.lstm_size = lstm_size
        self.lstm_depth = lstm_depth
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_size, num_layers=lstm_depth, batch_first=True, dropout=dropout if dropout else 0)
        self.fc1 = nn.Linear(lstm_size, self.input_size)
        if dropout is not None:
            self.dropout_layer = nn.Dropout(p=dropout)

        self._features = features
        self._features_stats = None
        self._hist_train_loss = []
        self._hist_train_loss_num_cdms = []

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self._hist_train_loss_num_cdms, self._hist_train_loss, label='Training')
        ax.set_xlabel('Training CDMs')
        ax.set_ylabel('Loss')
        ax.legend()

    def learn(self, event_set, lr, epochs, batch_size, device, valid_proportion=0.15, num_workers=4):
        if device is None:
            device = torch.device('cpu')

        num_params = sum(p.numel() for p in self.parameters())
        print('LSTM predictor with params: {}'.format(num_params))

        if self._features_stats is None:
            print('Computing feature statistics')
            self._features_stats = DatasetEventSet(event_set, self._features)._features_stats

        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        event_set = event_set.filter(lambda event: len(event) > 1)

        valid_set_size = int(len(event_set) * valid_proportion)
        train_set_size = len(event_set) - valid_set_size
        train_set = DatasetEventSet(event_set[:train_set_size], self._features, self._features_stats)
        valid_set = DatasetEventSet(event_set[train_set_size:], self._features, self._features_stats)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=True, num_workers=num_workers)
        self.train()
        if len(self._hist_train_loss_num_cdms) == 0:
            num_cdms = 0
        else:
            num_cdms = self._hist_train_loss_num_cdms[-1]
        for epoch in range(epochs):
            for i_minibatch, (events, event_lengths) in enumerate(train_loader):
                events, event_lengths = events.to(device), event_lengths.to(device)
                batch_size = event_lengths.nelement()  # Can be smaller than batch_size for the last minibatch of an epoch
                input = events[:, :-1]
                target = events[:, 1:]
                event_lengths -= 1
                self.reset(batch_size)
                optimizer.zero_grad()
                output = self(input, event_lengths)
                loss = nn.functional.mse_loss(output, target)
                loss.backward()
                optimizer.step()

                num_cdms += batch_size
                train_loss = float(loss)
                self._hist_train_loss_num_cdms.append(num_cdms)
                self._hist_train_loss.append(train_loss)
                print(train_loss)
        return self

    def predict(self, event):
        from .cdm import ConjunctionDataMessage
        from .event import EventSet

        self.to('cpu')
        ds = DatasetEventSet(EventSet(events=[event]), features=self._features)
        input, input_length = ds[0]
        self.train()
        self.reset(1)
        output = self.forward(input.unsqueeze(0), input_length.unsqueeze(0)).squeeze()
        if util.has_nan_or_inf(output):
            raise RuntimeError('Network output has nan or inf:\n'.format(output))
        if output.ndim == 1:
            output_last = output
        else:
            output_last = output[-1]

        date0 = event[0]['CREATION_DATE']
        cdm = ConjunctionDataMessage()
        for i in range(len(self._features)):
            feature = self._features[i]
            value = self._features_stats['mean'][i] + float(output_last[i].item()) * self._features_stats['stddev'][i]
            if feature == '__CREATION_DATE':
                cdm['CREATION_DATE'] = util.add_days_to_date_str(date0, value)
            elif feature == '__TCA':
                cdm['TCA'] = util.add_days_to_date_str(date0, value)
            else:
                cdm[feature] = value
        return cdm

    def reset(self, batch_size):
        h = torch.zeros(self.lstm_depth, batch_size, self.lstm_size)
        c = torch.zeros(self.lstm_depth, batch_size, self.lstm_size)
        device = list(self.parameters())[0].device
        h = h.to(device)
        c = c.to(device)
        self.hidden = (h, c)

    def forward(self, x, x_lengths):
        batch_size, x_length_max, _ = x.size()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        x, self.hidden = self.lstm(x, self.hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=x_length_max)
        if self.dropout:
            x = self.dropout_layer(x)
        x = torch.relu(x)
        x = self.fc1(x)
        return x
