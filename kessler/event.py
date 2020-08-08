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
        list_cdms = []
        for cdm in self._cdms:
            list_cdms.append(cdm.to_dataframe())
        return pd.concat(list_cdms, ignore_index=True)

    def plot_feature(self, feature_name, ax=None, *args, **kwargs):
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
        ax.plot(data_x, data_y)
        ax.scatter(data_x, data_y, *args, **kwargs)
        # ax.invert_xaxis()
        ax.set_xlim(max(data_x), min(data_x))
        ax.set_xlabel('Time to TCA')
        ax.set_title(feature_name)

    def plot_features(self, features, figsize=(10, 10)):
        rows, cols = util.tile_rows_cols(len(features))
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)

        for i, ax in enumerate(axs.flat):
            self.plot_feature(features[i], ax=ax)
            # ax.set_title(item)
        plt.tight_layout()

    def plot_uncertainty(self, figsize=(20, 10)):
        covariance_features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, covariance_features)) + list(map(lambda f: 'OBJECT2_'+f, covariance_features))
        self.plot_features(features, figsize=figsize)

    def __repr__(self):
        return 'Event(CDMs: {})'.format(len(self))

    def __getitem__(self, i):
        return self._cdms[i]

    def __len__(self):
        return len(self._cdms)


class EventCollection():
    def __init__(self, cdms_dir=None, cdm_extension='.cdm.kvn.txt'):
        self._events = []
        if cdms_dir is not None:
            print('Loading CDMS (with extension {}) from directory: {}'.format(cdm_extension, cdms_dir))
            file_names = sorted(glob(os.path.join(cdms_dir, '*' + cdm_extension)))
            regex = r"(.*)_([0-9]+{})".format(cdm_extension)
            matches = re.finditer(regex, '\n'.join(file_names))

            event_prefixes = []
            for m in matches:
                m = m.groups()[0]
                event_prefixes.append(m)
            event_prefixes = set(event_prefixes)
            len(event_prefixes)

            event_file_names = []
            for event_prefix in event_prefixes:
                event_file_names.append(list(filter(lambda f: f.startswith(event_prefix), file_names)))

            self._events = [Event(cdm_file_names=f) for f in event_file_names]
            print('Loaded {} CDMs grouped into {} events'.format(len(file_names), len(self._events)))

    def __getitem__(self, i):
        return self._events[i]

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        return 'EventCollection({})'.format(self._events)
