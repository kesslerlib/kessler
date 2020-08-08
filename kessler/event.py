import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import util
from .cdm import ConjunctionDataMessage

mpl.rcParams['axes.unicode_minus'] = False
plt_default_backend = plt.get_backend()


class Event():
    def __init__(self, cdms=None):
        if cdms:
            self._cdms = cdms
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
        ax.scatter(data_x, data_y, *args, **kwargs)
        # ax.invert_xaxis()
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
