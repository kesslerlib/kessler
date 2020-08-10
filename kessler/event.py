import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
import os
import re
import sys
from datetime import datetime, timedelta

from . import util
from .cdm import ConjunctionDataMessage
from .nn import LSTMPredictor

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
        self._update_cdm_extra_features()

    def _update_cdm_extra_features(self):
        if len(self._cdms) > 0:
            date0 = self._cdms[0]['CREATION_DATE']
            for cdm in self._cdms:
                cdm._values_extra['__CREATION_DATE'] = util.from_date_str_to_days(cdm['CREATION_DATE'], date0=date0)
                cdm._values_extra['__TCA'] = util.from_date_str_to_days(cdm['TCA'], date0=date0)

    def add(self, cdm):
        if isinstance(cdm, ConjunctionDataMessage):
            self._cdms.append(cdm)
        elif isinstance(cdm, list):
            for c in cdm:
                self.add(c)
        else:
            raise ValueError('Expecting a single CDM or a list of CDMs')
        self._update_cdm_extra_features()

    def to_dataframe(self):
        if len(self) == 0:
            return pd.DataFrame()
        cdm_dataframes = []
        for cdm in self._cdms:
            cdm_dataframes.append(cdm.to_dataframe())
        return pd.concat(cdm_dataframes, ignore_index=True)

    def plot_feature(self, feature_name, figsize=None, ax=None, return_ax=False, apply_func=None, file_name=None, legend=False, xlim=(-0.01, 7.01), ylims=None, *args, **kwargs):
        if apply_func is None:
            apply_func = lambda x: x
        data_x = []
        data_y = []
        for i, cdm in enumerate(self._cdms):
            if cdm['TCA'] is None:
                raise RuntimeError('CDM {} in event does not have TCA'.format(i))
            if cdm['CREATION_DATE'] is None:
                raise RuntimeError('CDM {} in event does not have CREATION_DATE'.format(i))
            time_to_tca = util.from_date_str_to_days(cdm['TCA'], date0=cdm['CREATION_DATE'])
            data_x.append(time_to_tca)
            data_y.append(apply_func(cdm[feature_name]))
        # Creating axes instance
        if ax is None:
            if figsize is None:
                figsize = 5, 3
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data_x, data_y, marker='.', *args, **kwargs)
        # ax.scatter(data_x, data_y)
        ax.set_xlabel('Time to TCA')
        ax.set_title(feature_name)

        # xmin, xmax = min(ax.get_xlim()), max(ax.get_xlim())
        # ax.set_xlim(xmax, xmin)
        if xlim is not None:
            xmin, xmax = xlim
            ax.set_xlim(xmax, xmin)

        if ylims is not None:
            if feature_name in ylims:
                ymin, ymax = ylims[feature_name]
                ax.set_ylim(ymin, ymax)

        if legend:
            ax.legend()

        if file_name is not None:
            plt.savefig(file_name)

        if return_ax:
            return ax

    def plot_features(self, feature_names, figsize=None, axs=None, return_axs=False, file_name=None, *args, **kwargs):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        rows, cols = util.tile_rows_cols(len(feature_names))
        if axs is None:
            if figsize is None:
                figsize = (cols*20/7, rows*12/6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)

        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                if i != 0 and 'legend' in kwargs:
                    kwargs['legend'] = False

                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize=(20, 12), diagonal=True, *args, **kwargs):
        if diagonal:
            features = ['CR_R', 'CT_T', 'CN_N', 'CRDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_NDOT']
        else:
            features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, features)) + list(map(lambda f: 'OBJECT2_'+f, features))
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

    def plot_feature(self, feature_name, figsize=None, ax=None, return_ax=False, file_name=None, *args, **kwargs):
        if ax is None:
            if figsize is None:
                figsize = 5, 3
            fig, ax = plt.subplots(figsize=figsize)
        for event in self:
            event.plot_feature(feature_name, ax=ax, *args, **kwargs)
            if 'label' in kwargs:
                kwargs.pop('label')  # We want to label only the first Event in this EventSet, for not cluttering the legend

        if file_name is not None:
            plt.savefig(file_name)

        if return_ax:
            return ax

    def plot_features(self, feature_names, figsize=None, axs=None, return_axs=False, file_name=None, *args, **kwargs):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        if axs is None:
            rows, cols = util.tile_rows_cols(len(feature_names))
            if figsize is None:
                figsize = (cols*20/7, rows*12/6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)

        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                if i != 0 and 'legend' in kwargs:
                    kwargs['legend'] = False

                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_name)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize=(20, 12), diagonal=True, *args, **kwargs):
        if diagonal:
            features = ['CR_R', 'CT_T', 'CN_N', 'CRDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_NDOT']
        else:
            features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, features)) + list(map(lambda f: 'OBJECT2_'+f, features))
        return self.plot_features(features, figsize=figsize, *args, **kwargs)

    def train_predictor(self, features=None, device=None, lr=1e-3, epochs=10, batch_size=16, dropout=0.2, model=None):
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
        if model is None:
            model = LSTMPredictor(features=features, dropout=dropout, event_set=self)
        model.learn(lr=lr, epochs=epochs, batch_size=batch_size, event_set=self, device=device)
        return model

    def filter(self, filter_func):
        events = []
        for event in self:
            if filter_func(event):
                events.append(event)
        return EventSet(events=events)

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


def kelvins_to_events(file_name, num_events=None, date_tca=None, remove_outliers=True, drop_features=['c_rcs_estimate', 't_rcs_estimate']):
    print('Loading Kelvins dataset from file name: {}'.format(file_name))
    kelvins = pd.read_csv(file_name)
    print('{} entries'.format(len(kelvins)))
    print('Dropping features: {}'.format(drop_features))
    kelvins = kelvins.drop(drop_features, axis=1)
    print('Dropping rows with NaNs')
    kelvins = kelvins.dropna()
    print('{} entries'.format(len(kelvins)))

    if remove_outliers:
        # outlier_features = ['CR_R', 'CT_T', 'CN_N', 'CRDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_NDOT']
        # outlier_features = ['t_sigma_r', 't_sigma_t', 't_sigma_n', 't_sigma_rdot', 't_sigma_tdot', 't_sigma_ndot']
        print('Removing outliers')
        kelvins = kelvins[kelvins['t_sigma_r'] <= 20]
        kelvins = kelvins[kelvins['c_sigma_r'] <= 1000]
        kelvins = kelvins[kelvins['t_sigma_t'] <= 2000]
        kelvins = kelvins[kelvins['c_sigma_t'] <= 100000]
        kelvins = kelvins[kelvins['t_sigma_n'] <= 10]
        kelvins = kelvins[kelvins['c_sigma_n'] <= 450]

        # for feature in outlier_features:
        #     # Q1 = kelvins[feature].quantile(0.25)
        #     # Q3 = kelvins[feature].quantile(0.75)
        #     # IQR = Q3 - Q1
        #     # limit = 1.5 * IQR
        #     # kelvins = kelvins[~((kelvins[feature] < (Q1 - limit)) | (kelvins[feature] > (Q3 + limit)))]
        #     kelvins = kelvins[kelvins[feature].between(kelvins[feature].quantile(.001), kelvins[feature].quantile(.75))]  # without outliers
        # kelvins = kelvins.reset_index()
        print('{} entries'.format(len(kelvins)))

    print('Shuffling')
    kelvins = kelvins.sample(frac=1, axis=1).reset_index(drop=True)
    kelvins_events = kelvins.groupby('event_id').groups
    print('Grouped rows into {} events'.format(len(kelvins_events)))
    if date_tca is None:
        date_tca = datetime.now()
    print('Taking TCA as current time: {}'.format(date_tca))
    events = []
    if num_events is None:
        num_events = len(kelvins_events)
    num_events = min(num_events, len(kelvins_events))
    i = 0
    for k, v in kelvins_events.items():
        i += 1
        if i > num_events:
            break
        print('Converting event {} / {}'.format(i, num_events), end='\r')
        sys.stdout.flush()
        kelvins_event = kelvins.iloc[v]
        cdms = []
        for _, kelvins_cdm in kelvins_event.iterrows():
            cdm = ConjunctionDataMessage()
            time_to_tca = kelvins_cdm['time_to_tca']  # days
            date_creation = date_tca - timedelta(days=time_to_tca)
            cdm['CREATION_DATE'] = util.from_datetime_to_cdm_datetime_str(date_creation)
            cdm['TCA'] = util.from_datetime_to_cdm_datetime_str(date_tca)
            cdm['MISS_DISTANCE'] = kelvins_cdm['miss_distance']
            cdm['RELATIVE_SPEED'] = kelvins_cdm['relative_speed']
            cdm['RELATIVE_POSITION_R'] = kelvins_cdm['relative_position_r']
            cdm['RELATIVE_POSITION_T'] = kelvins_cdm['relative_position_t']
            cdm['RELATIVE_POSITION_N'] = kelvins_cdm['relative_position_n']
            cdm['RELATIVE_VELOCITY_R'] = kelvins_cdm['relative_velocity_r']
            cdm['RELATIVE_VELOCITY_T'] = kelvins_cdm['relative_velocity_t']
            cdm['RELATIVE_VELOCITY_N'] = kelvins_cdm['relative_velocity_n']
            cdm['OBJECT1_CR_R'] = kelvins_cdm['t_sigma_r']**2.
            cdm['OBJECT1_CT_R'] = kelvins_cdm['t_ct_r']
            cdm['OBJECT1_CT_T'] = kelvins_cdm['t_sigma_t']**2.
            cdm['OBJECT1_CN_R'] = kelvins_cdm['t_cn_r']
            cdm['OBJECT1_CN_T'] = kelvins_cdm['t_cn_t']
            cdm['OBJECT1_CN_N'] = kelvins_cdm['t_sigma_n']**2.
            cdm['OBJECT1_CRDOT_R'] = kelvins_cdm['t_crdot_r']
            cdm['OBJECT1_CRDOT_T'] = kelvins_cdm['t_crdot_t']
            cdm['OBJECT1_CRDOT_N'] = kelvins_cdm['t_crdot_n']
            cdm['OBJECT1_CRDOT_RDOT'] = kelvins_cdm['t_sigma_rdot']**2.
            cdm['OBJECT1_CTDOT_R'] = kelvins_cdm['t_ctdot_r']
            cdm['OBJECT1_CTDOT_T'] = kelvins_cdm['t_ctdot_t']
            cdm['OBJECT1_CTDOT_N'] = kelvins_cdm['t_ctdot_n']
            cdm['OBJECT1_CTDOT_RDOT'] = kelvins_cdm['t_ctdot_rdot']
            cdm['OBJECT1_CTDOT_TDOT'] = kelvins_cdm['t_sigma_tdot']**2.
            cdm['OBJECT1_CNDOT_R'] = kelvins_cdm['t_cndot_r']
            cdm['OBJECT1_CNDOT_T'] = kelvins_cdm['t_cndot_t']
            cdm['OBJECT1_CNDOT_N'] = kelvins_cdm['t_cndot_n']
            cdm['OBJECT1_CNDOT_RDOT'] = kelvins_cdm['t_cndot_rdot']
            cdm['OBJECT1_CNDOT_TDOT'] = kelvins_cdm['t_cndot_tdot']
            cdm['OBJECT1_CNDOT_NDOT'] = kelvins_cdm['t_sigma_ndot']**2.

            cdm['OBJECT1_RECOMMENDED_OD_SPAN'] = kelvins_cdm['t_recommended_od_span']
            cdm['OBJECT1_ACTUAL_OD_SPAN'] = kelvins_cdm['t_actual_od_span']
            cdm['OBJECT1_OBS_AVAILABLE'] = kelvins_cdm['t_obs_available']
            cdm['OBJECT1_OBS_USED'] = kelvins_cdm['t_obs_used']
            cdm['OBJECT1_RESIDUALS_ACCEPTED'] = kelvins_cdm['t_residuals_accepted']
            cdm['OBJECT1_WEIGHTED_RMS'] = kelvins_cdm['t_weighted_rms']
            cdm['OBJECT1_SEDR'] = kelvins_cdm['t_sedr']
            time_lastob_start = kelvins_cdm['t_time_lastob_start']  # days until CDM creation
            time_lastob_start = date_creation - timedelta(days=time_lastob_start)
            cdm['OBJECT1_TIME_LASTOB_START'] = util.from_datetime_to_cdm_datetime_str(time_lastob_start)
            time_lastob_end = kelvins_cdm['t_time_lastob_end']  # days until CDM creation
            time_lastob_end = date_creation - timedelta(days=time_lastob_end)
            cdm['OBJECT1_TIME_LASTOB_END'] = util.from_datetime_to_cdm_datetime_str(time_lastob_end)

            cdm['OBJECT2_CR_R'] = kelvins_cdm['c_sigma_r']**2.
            cdm['OBJECT2_CT_R'] = kelvins_cdm['c_ct_r']
            cdm['OBJECT2_CT_T'] = kelvins_cdm['c_sigma_t']**2.
            cdm['OBJECT2_CN_R'] = kelvins_cdm['c_cn_r']
            cdm['OBJECT2_CN_T'] = kelvins_cdm['c_cn_t']
            cdm['OBJECT2_CN_N'] = kelvins_cdm['c_sigma_n']**2.
            cdm['OBJECT2_CRDOT_R'] = kelvins_cdm['c_crdot_r']
            cdm['OBJECT2_CRDOT_T'] = kelvins_cdm['c_crdot_t']
            cdm['OBJECT2_CRDOT_N'] = kelvins_cdm['c_crdot_n']
            cdm['OBJECT2_CRDOT_RDOT'] = kelvins_cdm['c_sigma_rdot']**2.
            cdm['OBJECT2_CTDOT_R'] = kelvins_cdm['c_ctdot_r']
            cdm['OBJECT2_CTDOT_T'] = kelvins_cdm['c_ctdot_t']
            cdm['OBJECT2_CTDOT_N'] = kelvins_cdm['c_ctdot_n']
            cdm['OBJECT2_CTDOT_RDOT'] = kelvins_cdm['c_ctdot_rdot']
            cdm['OBJECT2_CTDOT_TDOT'] = kelvins_cdm['c_sigma_tdot']**2.
            cdm['OBJECT2_CNDOT_R'] = kelvins_cdm['c_cndot_r']
            cdm['OBJECT2_CNDOT_T'] = kelvins_cdm['c_cndot_t']
            cdm['OBJECT2_CNDOT_N'] = kelvins_cdm['c_cndot_n']
            cdm['OBJECT2_CNDOT_RDOT'] = kelvins_cdm['c_cndot_rdot']
            cdm['OBJECT2_CNDOT_TDOT'] = kelvins_cdm['c_cndot_tdot']
            cdm['OBJECT2_CNDOT_NDOT'] = kelvins_cdm['c_sigma_ndot']**2.

            cdm['OBJECT2_OBJECT_TYPE'] = kelvins_cdm['c_object_type']
            cdm['OBJECT2_RECOMMENDED_OD_SPAN'] = kelvins_cdm['c_recommended_od_span']
            cdm['OBJECT2_ACTUAL_OD_SPAN'] = kelvins_cdm['c_actual_od_span']
            cdm['OBJECT2_OBS_AVAILABLE'] = kelvins_cdm['c_obs_available']
            cdm['OBJECT2_OBS_USED'] = kelvins_cdm['c_obs_used']
            cdm['OBJECT2_RESIDUALS_ACCEPTED'] = kelvins_cdm['c_residuals_accepted']
            cdm['OBJECT2_WEIGHTED_RMS'] = kelvins_cdm['c_weighted_rms']
            cdm['OBJECT2_SEDR'] = kelvins_cdm['c_sedr']
            time_lastob_start = kelvins_cdm['c_time_lastob_start']  # days until CDM creation
            time_lastob_start = date_creation - timedelta(days=time_lastob_start)
            cdm['OBJECT2_TIME_LASTOB_START'] = util.from_datetime_to_cdm_datetime_str(time_lastob_start)
            time_lastob_end = kelvins_cdm['c_time_lastob_end']  # days until CDM creation
            time_lastob_end = date_creation - timedelta(days=time_lastob_end)
            cdm['OBJECT2_TIME_LASTOB_END'] = util.from_datetime_to_cdm_datetime_str(time_lastob_end)

            cdms.append(cdm)
        events.append(Event(cdms))

    return EventSet(events=events)
