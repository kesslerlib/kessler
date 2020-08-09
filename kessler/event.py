import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
import os
import re

from . import util
from .cdm import ConjunctionDataMessage

mpl.rcParams['axes.unicode_minus'] = False
plt_default_backend = plt.get_backend()


class Event():
    def __init__(self, cdms=None, cdm_file_names=None):
        if cdms is not None:
            if cdm_file_names is not None:
                raise RuntimeError('Expecting only one of cdms, cdm_file_names, not both')
            self._cdms = cdms
        elif cdm_file_names is not None:
            self._cdms = [ConjunctionDataMessage(file_name) for file_name in cdm_file_names]
        else:
            self._cdms = []

    def add(self, cdm):
        if isinstance(cdm, ConjunctionDataMessage):
            self._cdms.append(cdm)
        elif isinstance(cdm, list):
            for c in cdm:
                self.add(c)
        else:
            raise ValueError('Expecting a single CDM or a list of CDMs')

    def to_dataframe(self):
        if len(self) == 0:
            return pd.DataFrame()
        cdm_dataframes = []
        for cdm in self._cdms:
            cdm_dataframes.append(cdm.to_dataframe())
        return pd.concat(cdm_dataframes, ignore_index=True)

    def plot_feature(self, feature_name, ax=None, return_ax=False, other_events=None, legend=True, label=None, *args, **kwargs):
        data_x = []
        data_y = []
        for i, cdm in enumerate(self._cdms):
            if cdm['TCA'] is None:
                raise RuntimeError('CDM {} in event does not have TCA'.format(i))
            if cdm['CREATION_DATE'] is None:
                raise RuntimeError('CDM {} in event does not have CREATION_DATE'.format(i))
            data_x.append(util.from_date_str_to_days(cdm['TCA'], cdm['CREATION_DATE']))
            data_y.append(cdm[feature_name])
        # Creating axes instance
        if ax is None:
            fig, ax = plt.subplots()
        if label is None:
            label = 'Event 0'
        ax.plot(data_x, data_y, marker='.', label=label, *args, **kwargs)
        ax.set_xlabel('Time to TCA')
        ax.set_title(feature_name)
        xmin, xmax = min(ax.get_xlim()), max(ax.get_xlim())

        if other_events is not None:
            if not isinstance(other_events, list):
                if isinstance(other_events, EventCollection):
                    other_events = list(other_events)
                elif isinstance(other_events, Event):
                    other_events = [other_events]
                else:
                    raise ValueError('Expecting other_events to be one of (Event, EventCollection, or a list of Events)')
            for i, e in enumerate(other_events):
                eax = e.plot_feature(feature_name, ax=ax, return_ax=True, label='Event {}'.format(i+1), *args, **kwargs)
                exmin, exmax = min(eax.get_xlim()), max(eax.get_xlim())
                xmin, xmax = min(xmin, exmin), max(xmax, exmax)
            if legend:
                ax.legend()
        ax.set_xlim(xmax, xmin)

        if return_ax:
            return ax

    def plot_features(self, features, figsize=None, return_ax=False, other_events=None, legend=True, *args, **kwargs):
        if not isinstance(features, list):
            features = [features]
        rows, cols = util.tile_rows_cols(len(features))
        if figsize is None:
            figsize = (cols*20/7, rows*12/6)
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)

        for i, ax in enumerate(axs.flat):
            if i < len(features):
                self.plot_feature(features[i], ax=ax, other_events=other_events, legend=legend, *args, **kwargs)
                if i != 0 and ax.legend_ is not None:
                    ax.legend_.remove()
            else:
                ax.axis('off')
        plt.tight_layout()

        if return_ax:
            return axs

    def plot_uncertainty(self, figsize=(20, 12), *args, **kwargs):
        covariance_features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, covariance_features)) + list(map(lambda f: 'OBJECT2_'+f, covariance_features))
        return self.plot_features(features, figsize=figsize, *args, **kwargs)

    def __repr__(self):
        return 'Event(CDMs: {})'.format(len(self))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Event(cdms=self._cdms[index])
        else:
            return self._cdms[index]

    def __len__(self):
        return len(self._cdms)


class EventCollection():
    def __init__(self, cdms_dir=None, cdm_extension='.cdm.kvn.txt', events=None):
        if events is None:
            if cdms_dir is None:
                self._events = []
            else:
                print('Loading CDMS (with extension {}) from directory: {}'.format(cdm_extension, cdms_dir))
                file_names = sorted(glob(os.path.join(cdms_dir, '*' + cdm_extension)))
                regex = r"(.*)_([0-9]+{})".format(cdm_extension)
                matches = re.finditer(regex, '\n'.join(file_names))

                event_prefixes = []
                for m in matches:
                    m = m.groups()[0]
                    event_prefixes.append(m)
                event_prefixes = sorted(set(event_prefixes))

                event_file_names = []
                for event_prefix in event_prefixes:
                    event_file_names.append(list(filter(lambda f: f.startswith(event_prefix), file_names)))

                self._events = [Event(cdm_file_names=f) for f in event_file_names]
                print('Loaded {} CDMs grouped into {} events'.format(len(file_names), len(self._events)))
        else:
            self._events = events

    def to_dataframe(self):
        if len(self) == 0:
            return pd.DataFrame()
        event_dataframes = []
        for event in self._events:
            event_dataframes.append(event.to_dataframe())
        return pd.concat(event_dataframes, ignore_index=True)

    def plot_feature(self, feature_name, ax=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        for event in self:
            event.plot_feature(feature_name, ax=ax, *args, **kwargs)
        plt.tight_layout()

    def plot_features(self, features, figsize=None, *args, **kwargs):
        if not isinstance(features, list):
            features = [features]
        rows, cols = util.tile_rows_cols(len(features))
        if figsize is None:
            figsize = (cols*20/7, rows*12/6)
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)

        for i, ax in enumerate(axs.flat):
            if i < len(features):
                self.plot_feature(features[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

    def plot_uncertainty(self, figsize=(20, 12), *args, **kwargs):
        covariance_features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, covariance_features)) + list(map(lambda f: 'OBJECT2_'+f, covariance_features))
        return self.plot_features(features, figsize=figsize, *args, **kwargs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return EventCollection(events=self._events[index])
        else:
            return self._events[index]

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        if len(self) == 0:
            return 'EventCollection()'
        else:
            event_lengths = list(map(len, self._events))
            event_lengths_min = min(event_lengths)
            event_lengths_max = max(event_lengths)
            event_lengths_mean = sum(event_lengths)/len(event_lengths)
            return 'EventCollection(Events:{}, number of CDMs per event: {} (min), {} (max), {:.2f} (mean))'.format(len(self._events), event_lengths_min, event_lengths_max, event_lengths_mean)
