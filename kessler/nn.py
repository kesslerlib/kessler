import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from . import util
from .cdm import ConjunctionDataMessage
from .event import EventDataset


class DatasetEventDataset(Dataset):
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
    def __init__(self, lstm_size=256, lstm_depth=2, dropout=0.2, features=None):
        super().__init__()
        if features is None:
            features = ['__CREATION_DATE',
                        '__TCA',
                        'MISS_DISTANCE',
                        'RELATIVE_SPEED',
                        'RELATIVE_POSITION_R',
                        'RELATIVE_POSITION_T',
                        'RELATIVE_POSITION_N',
                        'RELATIVE_VELOCITY_R',
                        'RELATIVE_VELOCITY_T',
                        'RELATIVE_VELOCITY_N',
                        'OBJECT1_X',
                        'OBJECT1_Y',
                        'OBJECT1_Z',
                        'OBJECT1_X_DOT',
                        'OBJECT1_Y_DOT',
                        'OBJECT1_Z_DOT',
                        'OBJECT1_CR_R',
                        'OBJECT1_CT_R',
                        'OBJECT1_CT_T',
                        'OBJECT1_CN_R',
                        'OBJECT1_CN_T',
                        'OBJECT1_CN_N',
                        'OBJECT1_CRDOT_R',
                        'OBJECT1_CRDOT_T',
                        'OBJECT1_CRDOT_N',
                        'OBJECT1_CRDOT_RDOT',
                        'OBJECT1_CTDOT_R',
                        'OBJECT1_CTDOT_T',
                        'OBJECT1_CTDOT_N',
                        'OBJECT1_CTDOT_RDOT',
                        'OBJECT1_CTDOT_TDOT',
                        'OBJECT1_CNDOT_R',
                        'OBJECT1_CNDOT_T',
                        'OBJECT1_CNDOT_N',
                        'OBJECT1_CNDOT_RDOT',
                        'OBJECT1_CNDOT_TDOT',
                        'OBJECT1_CNDOT_NDOT',
                        'OBJECT2_X',
                        'OBJECT2_Y',
                        'OBJECT2_Z',
                        'OBJECT2_X_DOT',
                        'OBJECT2_Y_DOT',
                        'OBJECT2_Z_DOT',
                        'OBJECT2_CR_R',
                        'OBJECT2_CT_R',
                        'OBJECT2_CT_T',
                        'OBJECT2_CN_R',
                        'OBJECT2_CN_T',
                        'OBJECT2_CN_N',
                        'OBJECT2_CRDOT_R',
                        'OBJECT2_CRDOT_T',
                        'OBJECT2_CRDOT_N',
                        'OBJECT2_CRDOT_RDOT',
                        'OBJECT2_CTDOT_R',
                        'OBJECT2_CTDOT_T',
                        'OBJECT2_CTDOT_N',
                        'OBJECT2_CTDOT_RDOT',
                        'OBJECT2_CTDOT_TDOT',
                        'OBJECT2_CNDOT_R',
                        'OBJECT2_CNDOT_T',
                        'OBJECT2_CNDOT_N',
                        'OBJECT2_CNDOT_RDOT',
                        'OBJECT2_CNDOT_TDOT',
                        'OBJECT2_CNDOT_NDOT']

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
        self._hist_train_loss_iters = []
        self._hist_valid_loss = []
        self._hist_valid_loss_iters = []

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self._hist_train_loss_iters, self._hist_train_loss, label='Training')
        ax.plot(self._hist_valid_loss_iters, self._hist_valid_loss, label='Validation')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()

    def learn(self, event_set, epochs=2, lr=1e-3, batch_size=8, device='cpu', valid_proportion=0.15, num_workers=4):
        if device is None:
            device = torch.device('cpu')

        num_params = sum(p.numel() for p in self.parameters())
        print('LSTM predictor with params: {:,}'.format(num_params))

        if self._features_stats is None:
            print('Computing feature statistics')
            self._features_stats = DatasetEventDataset(event_set, self._features)._features_stats

        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        event_set = event_set.filter(lambda event: len(event) > 1)

        valid_set_size = int(len(event_set) * valid_proportion)
        train_set_size = len(event_set) - valid_set_size
        train_set = DatasetEventDataset(event_set[:train_set_size], self._features, self._features_stats)
        valid_set = DatasetEventDataset(event_set[train_set_size:], self._features, self._features_stats)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=True, num_workers=num_workers)
        self.train()
        if len(self._hist_train_loss_iters) == 0:
            total_iters = 0
        else:
            total_iters = self._hist_train_loss_iters[-1]
        for epoch in range(epochs):
            with torch.no_grad():
                for _, (events, event_lengths) in enumerate(valid_loader):
                    events, event_lengths = events.to(device), event_lengths.to(device)
                    batch_size = event_lengths.nelement()  # Can be smaller than batch_size for the last minibatch of an epoch
                    input = events[:, :-1]
                    target = events[:, 1:]
                    event_lengths -= 1
                    self.reset(batch_size)
                    output = self(input, event_lengths)
                    loss = nn.functional.mse_loss(output, target)
                    valid_loss = float(loss)
                    self._hist_valid_loss_iters.append(total_iters)
                    self._hist_valid_loss.append(valid_loss)

            for i_minibatch, (events, event_lengths) in enumerate(train_loader):
                total_iters += 1
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

                train_loss = float(loss)
                self._hist_train_loss_iters.append(total_iters)
                self._hist_train_loss.append(train_loss)

                print('{} | {}/{} | {:.4e} | {:.4e}'.format(total_iters, epoch+1, epochs, train_loss, valid_loss), end='\r')

        return self

    def predict(self, event):
        ds = DatasetEventDataset(EventDataset(events=[event]), features=self._features, features_stats=self._features_stats)
        input, input_length = ds[0]
        device = list(self.parameters())[0].device
        input, input_length = input.to(device), input_length.to(device)
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
                if value < event[-1]['__CREATION_DATE']:
                    value = event[-1]['__CREATION_DATE']
                cdm['CREATION_DATE'] = util.add_days_to_date_str(date0, value)
            elif feature == '__TCA':
                cdm['TCA'] = util.add_days_to_date_str(date0, value)
            else:
                cdm[feature] = value
        return cdm

    def predict_event_step(self, event, num_samples=1):
        es = []
        for i in range(num_samples):
            e = event.copy()
            cdm = self.predict(e)
            e.add(cdm)
            es.append(e)
        if num_samples == 1:
            return es[0]
        else:
            return EventDataset(events=es)

    def predict_event(self, event, num_samples=1, max_length=22):
        es = []
        for i in range(num_samples):
            e = event.copy()
            while True:
                cdm = self.predict(e)
                e.add(cdm)
    #             print('Next cdm {}, {}'.format(cdm['__CREATION_DATE'], cdm['__TCA']))
                if cdm['__CREATION_DATE'] > cdm['__TCA']:
                    break
                if cdm['__TCA'] > 7:
                    break
                if len(e) > max_length:
                    # print('Max length ({}) reached'.format(max_length))
                    break
            es.append(e)
        if num_samples == 1:
            return es[0]
        else:
            return EventDataset(events=es)

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
