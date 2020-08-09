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

    def plot_feature(self, feature_name, figsize=None, ax=None, return_ax=False, *args, **kwargs):
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
            if figsize is None:
                figsize = 5, 3
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data_x, data_y, marker='.', *args, **kwargs)
        # ax.scatter(data_x, data_y)
        ax.set_xlabel('Time to TCA')
        ax.set_title(feature_name)
        xmin, xmax = min(ax.get_xlim()), max(ax.get_xlim())
        ax.set_xlim(xmax, xmin)

        if return_ax:
            return ax

    def plot_features(self, feature_names, figsize=None, axs=None, return_axs=False, *args, **kwargs):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        rows, cols = util.tile_rows_cols(len(feature_names))
        if axs is None:
            if figsize is None:
                figsize = (cols*20/7, rows*12/6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)

        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

        if return_axs:
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


class EventSet():
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

    def plot_feature(self, feature_name, figsize=None, ax=None, return_ax=False, *args, **kwargs):
        if ax is None:
            if figsize is None:
                figsize = 5, 3
            fig, ax = plt.subplots(figsize=figsize)
        for event in self:
            event.plot_feature(feature_name, ax=ax, *args, **kwargs)
            if 'label' in kwargs:
                kwargs.pop('label')  # We want to label only the first Event in this EventSet, for not cluttering the legend
        if return_ax:
            return ax

    def plot_features(self, feature_names, figsize=None, axs=None, return_axs=False, *args, **kwargs):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        if axs is None:
            rows, cols = util.tile_rows_cols(len(feature_names))
            if figsize is None:
                figsize = (cols*20/7, rows*12/6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)

        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize=(20, 12), *args, **kwargs):
        covariance_features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, covariance_features)) + list(map(lambda f: 'OBJECT2_'+f, covariance_features))
        return self.plot_features(features, figsize=figsize, *args, **kwargs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return EventSet(events=self._events[index])
        else:
            return self._events[index]

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        if len(self) == 0:
            return 'EventSet()'
        else:
            event_lengths = list(map(len, self._events))
            event_lengths_min = min(event_lengths)
            event_lengths_max = max(event_lengths)
            event_lengths_mean = sum(event_lengths)/len(event_lengths)
            return 'EventSet(Events:{}, number of CDMs per event: {} (min), {} (max), {:.2f} (mean))'.format(len(self._events), event_lengths_min, event_lengths_max, event_lengths_mean)
