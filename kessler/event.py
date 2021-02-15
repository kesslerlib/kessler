import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
import copy
import sys
import os
import re

from . import util
from .cdm import ConjunctionDataMessage
from .cdm import CDM

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
                cdm._values_extra['__DAYS_TO_TCA'] = cdm._values_extra['__TCA'] - cdm._values_extra['__CREATION_DATE']

    def add(self, cdm, return_result=False):
        if isinstance(cdm, ConjunctionDataMessage):
            self._cdms.append(cdm)
        elif isinstance(cdm, list):
            for c in cdm:
                self.add(c)
        else:
            raise ValueError('Expecting a single CDM or a list of CDMs')
        self._update_cdm_extra_features()
        if return_result:
            return self

    def copy(self):
        return Event(cdms=copy.deepcopy(self._cdms))

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
        ax.set_xlabel('Time to TCA (days)')
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
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

        if return_ax:
            return ax

    def plot_features(self, feature_names, figsize=None, axs=None, return_axs=False, file_name=None, sharex=True, *args, **kwargs):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        if axs is None:
            rows, cols = util.tile_rows_cols(len(feature_names))
            if figsize is None:
                figsize = (cols*20/7, rows*12/6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)

        if not isinstance(axs, np.ndarray):
            axs = np.array(axs)
        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                if i != 0 and 'legend' in kwargs:
                    kwargs['legend'] = False

                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize=(20, 12), diagonal=False, *args, **kwargs):
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


class EventDataset():
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

    # Proposed API for the pandas loader
    @staticmethod
    def from_pandas(df, cdm_compatible_fields={'RELATIVE_SPEED': 'relative_speed'}, group_events_by='EVENT_ID', object_1_prefix='T_', object_2_prefix='C_'):
        df.columns=[df.columns.tolist()[i].upper() for i in range(len(df.columns.tolist()))]

        df_events = df.groupby(group_events_by).groups
        num_events = len(df_events)
        column_not_present=[]
        column_not_present_counter=0
        events=[]
        i=0
        for k,v in df_events.items():
            i+=1
            print('Converting event {} / {}'.format(i, num_events), end='\r')
            sys.stdout.flush()
            df_event = df.iloc[v]
            cdms=[]
            column_not_present_counter=0
            for _, df_cdm in df_event.iterrows():
                cdm = CDM()
                for column in df.columns:
                    column_name=column[2:]
                    column_prefix=column[0:2]
                    if (column_name in cdm._keys_header or  column_name in cdm._keys_relative_metadata or column_name in cdm._keys_metadata or column_name in cdm._keys_data_od or column_name in cdm._keys_data_state or column_name in cdm._keys_data_covariance):
                        if column_prefix==object_1_prefix:
                            cdm['OBJECT1_'+column_name]=df_cdm[column]
                        elif column_prefix==object_2_prefix:
                            cdm['OBJECT2_'+column_name]=df_cdm[column]
                        else:
                            if column_not_present_counter==0:
                                column_not_present.append(column)

                    elif (column in cdm._keys_header or  column in cdm._keys_relative_metadata or column in cdm._keys_metadata or column in cdm._keys_data_od or column in cdm._keys_data_state or column in cdm._keys_data_covariance):
                        cdm[column]=df_cdm[column]
                    else:
                        if column=='JSPOC_PROBABILITY':
                            cdm['COLLISION_PROBABILITY']=df_cdm[column]
                        elif column=='T_SPAN':
                            cdm['OBJECT1_ACTUAL_OD_SPAN']=df_cdm[column]
                        elif column=='C_SPAN':
                            cdm['OBJECT2_ACTUAL_OD_SPAN']=df_cdm[column]
                        else:
                            if column_not_present_counter==0:
                                column_not_present.append(column)
                if column_not_present_counter==0:
                    column_not_present_counter+=1
                    print(f'The following columns are not present:{column_not_present}')
                cdms.append(cdm)
            events.append(Event(cdms))
        return EventDataset(events=events)

    def to_dataframe(self):
        if len(self) == 0:
            return pd.DataFrame()
        event_dataframes = []
        for event in self._events:
            event_dataframes.append(event.to_dataframe())
        return pd.concat(event_dataframes, ignore_index=True)

    def dates(self):
        print('CDM| CREATION_DATE (mean)       | Days (mean, std)  | Days to TCA (mean, std)')
        for i in range(self.event_lengths_max):
            creation_date_days = []
            days_to_tca = []
            for event in self:
                if i < len(event):
                    creation_date_days.append(event[i]['__CREATION_DATE'])
                    days_to_tca.append(event[i]['__DAYS_TO_TCA'])
            creation_date_days = np.array(creation_date_days)
            creation_date_days_mean, creation_date_days_stddev = creation_date_days.mean(), creation_date_days.std()
            days_to_tca = np.array(days_to_tca)
            days_to_tca_mean, days_to_tca_stddev = days_to_tca.mean(), days_to_tca.std()
            date0 = self[0][0]['CREATION_DATE']
            creation_date_days_mean_str = util.add_days_to_date_str(date0, creation_date_days_mean)
            print('{:02d} | {} | {:.6f} {:.6f} | {:.6f} {:.6f}'.format(i+1, creation_date_days_mean_str, creation_date_days_mean, creation_date_days_stddev, days_to_tca_mean, days_to_tca_stddev))

    @property
    def event_lengths(self):
        return list(map(len, self._events))

    @property
    def event_lengths_min(self):
        return min(self.event_lengths)

    @property
    def event_lengths_max(self):
        return max(self.event_lengths)

    @property
    def event_lengths_mean(self):
        return np.array(self.event_lengths).mean()

    @property
    def event_lengths_stddev(self):
        return np.array(self.event_lengths).std()

    def plot_event_lengths(self, figsize=(6, 4), file_name=None, *args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        event_lengths = self.event_lengths()
        ax.hist(event_lengths, *args, **kwargs)
        ax.set_xlabel('Event length (number of CDMs)')
        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

    def plot_feature(self, feature_name, figsize=None, ax=None, return_ax=False, file_name=None, *args, **kwargs):
        if ax is None:
            if figsize is None:
                figsize = 5, 3
            fig, ax = plt.subplots(figsize=figsize)
        for event in self:
            event.plot_feature(feature_name, ax=ax, *args, **kwargs)
            if 'label' in kwargs:
                kwargs.pop('label')  # We want to label only the first Event in this EventDataset, for not cluttering the legend

        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

        if return_ax:
            return ax

    def plot_features(self, feature_names, figsize=None, axs=None, return_axs=False, file_name=None, sharex=True, *args, **kwargs):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        if axs is None:
            rows, cols = util.tile_rows_cols(len(feature_names))
            if figsize is None:
                figsize = (cols*20/7, rows*12/6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)

        if not isinstance(axs, np.ndarray):
            axs = np.array(axs)
        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                if i != 0 and 'legend' in kwargs:
                    kwargs['legend'] = False

                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize=(20, 12), diagonal=False, *args, **kwargs):
        if diagonal:
            features = ['CR_R', 'CT_T', 'CN_N', 'CRDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_NDOT']
        else:
                features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, features)) + list(map(lambda f: 'OBJECT2_'+f, features))
        return self.plot_features(features, figsize=figsize, *args, **kwargs)

    def filter(self, filter_func):
        events = []
        for event in self:
            if filter_func(event):
                events.append(event)
        return EventDataset(events=events)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return EventDataset(events=self._events[index])
        else:
            return self._events[index]

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        if len(self) == 0:
            return 'EventDataset()'
        else:
            event_lengths = list(map(len, self._events))
            event_lengths_min = min(event_lengths)
            event_lengths_max = max(event_lengths)
            event_lengths_mean = sum(event_lengths)/len(event_lengths)
            return 'EventDataset(Events:{}, number of CDMs per event: {} (min), {} (max), {:.2f} (mean))'.format(len(self._events), event_lengths_min, event_lengths_max, event_lengths_mean)
