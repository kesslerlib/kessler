# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# Trillium Technologies
# University of Oxford
# Giacomo Acciarini (giacomo.acciarini@gmail.com)
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
import copy
import os
import re

from . import util
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
            self._cdms = [CDM(file_name) for file_name in cdm_file_names]
        else:
            self._cdms = []
        self._update_cdm_extra_features()
        self._dataframe = None

    def _update_cdm_extra_features(self):
        if len(self._cdms) > 0:
            date0 = self._cdms[0]['CREATION_DATE']
            for cdm in self._cdms:
                cdm._values_extra['__CREATION_DATE'] = util.from_date_str_to_days(cdm['CREATION_DATE'], date0=date0)
                cdm._values_extra['__TCA'] = util.from_date_str_to_days(cdm['TCA'], date0=date0)
                cdm._values_extra['__DAYS_TO_TCA'] = cdm._values_extra['__TCA'] - cdm._values_extra['__CREATION_DATE']

    def add(self, cdm, return_result=False):
        if isinstance(cdm, CDM):
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
        if self._dataframe is None:
            if len(self) == 0:
                self._dataframe = pd.DataFrame()
            cdm_dataframes = []
            for cdm in self._cdms:
                cdm_dataframes.append(cdm.to_dataframe())
            self._dataframe = pd.concat(cdm_dataframes, ignore_index=True)
        return self._dataframe

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
    def from_pandas(df, cdm_compatible_fields={
        'relative_speed': 'RELATIVE_SPEED',
        'ccsds_cdm_vers': 'CCSDS_CDM_VERS',
        'creation_date': 'CREATION_DATE',
        'originator':'ORIGINATOR',
        'message_for':'MESSAGE_FOR',
        'message_id':'MESSAGE_ID',
        'tca':'TCA',
        'miss_distance':'MISS_DISTANCE',
        'relative_speed':'RELATIVE_SPEED',
        'relative_position_r':'RELATIVE_POSITION_R',
        'relative_position_t':'RELATIVE_POSITION_T',
        'relative_position_n':'RELATIVE_POSITION_N',
        'relative_velocity_r':'RELATIVE_VELOCITY_R',
        'relative_velocity_t':'RELATIVE_VELOCITY_T',
        'relative_velocity_n':'RELATIVE_VELOCITY_N',
        'start_screen_period':'START_SCREEN_PERIOD',
        'stop_screen_period':'STOP_SCREEN_PERIOD',
        'screen_volume_frame':'SCREEN_VOLUME_FRAME',
        'screen_volume_shape':'SCREEN_VOLUME_SHAPE',
        'screen_volume_x':'SCREEN_VOLUME_X',
        'screen_volume_y':'SCREEN_VOLUME_Y',
        'screen_volume_z':'SCREEN_VOLUME_Z',
        'screen_entry_time':'SCREEN_ENTRY_TIME',
        'screen_exit_time':'SCREEN_EXIT_TIME',
        'jspoc_probability':'COLLISION_PROBABILITY',
        't_object_designator':'OBJECT1_OBJECT_DESIGNATOR',
        't_catalog_name':'OBJECT1_CATALOG_NAME',
        't_object_name':'OBJECT1_OBJECT_NAME',
        't_international_designator':'OBJECT1_INTERNATIONAL_DESIGNATOR',
        't_object_type':'OBJECT1_OBJECT_TYPE',
        't_ephemeris_name':'OBJECT1_EPHEMERIS_NAME',
        't_covariance_method':'OBJECT1_COVARIANCE_METHOD',
        't_maneuverable':'OBJECT1_MANEUVERABLE',
        't_orbit_center':'OBJECT1_ORBIT_CENTER',
        't_ref_frame':'OBJECT1_REF_FRAME',
        't_gravity_model':'OBJECT1_GRAVITY_MODEL',
        't_atmospheric_model':'OBJECT1_ATMOSPHERIC_MODEL',
        't_n_body_perturbations':'OBJECT1_N_BODY_PERTURBATIONS',
        't_solar_rad_pressure':'OBJECT1_SOLAR_RAD_PRESSURE',
        't_earth_tides':'OBJECT1_EARTH_TIDES',
        't_intrack_thrust':'OBJECT1_INTRACK_THRUST',
        't_time_lastob_start':'OBJECT1_TIME_LASTOB_START',
        't_time_lastob_end':'OBJECT1_TIME_LASTOB_END',
        't_recommended_od_span':'OBJECT1_RECOMMENDED_OD_SPAN',
        't_actual_od_span':'OBJECT1_ACTUAL_OD_SPAN',
        't_obs_available':'OBJECT1_OBS_AVAILABLE',
        't_obs_used':'OBJECT1_OBS_USED',
        't_tracks_available':'OBJECT1_TRACKS_AVAILABLE',
        't_tracks_used':'OBJECT1_TRACKS_USED',
        't_residuals_accepted':'OBJECT1_RESIDUALS_ACCEPTED',
        't_weighted_rms':'OBJECT1_WEIGHTED_RMS',
        't_area_pc':'OBJECT1_AREA_PC',
        't_area_drg':'OBJECT1_AREA_DRG',
        't_area_srg':'OBJECT1_AREA_SRP',
        't_mass':'OBJECT1_MASS',
        't_cd_area_over_mass':'OBJECT1_CD_AREA_OVER_MASS',
        't_cr_area_over_mass':'OBJECT1_CR_AREA_OVER_MASS',
        't_thrust_acceleration':'OBJECT1_THRUST_ACCELERATION',
        't_sedr':'OBJECT1_SEDR',
        't_x':'OBJECT1_X',
        't_y':'OBJECT1_Y',
        't_z':'OBJECT1_Z',
        't_x_dot':'OBJECT1_X_DOT',
        't_y_dot':'OBJECT1_Y_DOT',
        't_z_dot':'OBJECT1_Z_DOT',
        't_cr_r':'OBJECT1_CR_R',
        't_ct_r':'OBJECT1_CT_R',
        't_ct_t':'OBJECT1_CT_T',
        't_cn_r':'OBJECT1_CN_R',
        't_cn_t':'OBJECT1_CN_T',
        't_cn_n':'OBJECT1_CN_N',
        't_crdot_r':'OBJECT1_CRDOT_R',
        't_crdot_t':'OBJECT1_CRDOT_T',
        't_crdot_n':'OBJECT1_CRDOT_N',
        't_crdot_rdot':'OBJECT1_CRDOT_RDOT',
        't_ctdot_r':'OBJECT1_CTDOT_R',
        't_ctdot_t':'OBJECT1_CTDOT_T',
        't_ctdot_n':'OBJECT1_CTDOT_N',
        't_ctdot_rdot':'OBJECT1_CTDOT_RDOT',
        't_ctdot_tdot':'OBJECT1_CTDOT_TDOT',
        't_cndot_r':'OBJECT1_CNDOT_R',
        't_cndot_t':'OBJECT1_CNDOT_T',
        't_cndot_n':'OBJECT1_CNDOT_N',
        't_cndot_rdot':'OBJECT1_CNDOT_RDOT',
        't_cndot_tdot':'OBJECT1_CNDOT_TDOT',
        't_cndot_ndot':'OBJECT1_CNDOT_NDOT',
        't_cdrg_r':'OBJECT1_CDRG_R',
        't_cdrg_t':'OBJECT1_CDRG_T',
        't_cdrg_n':'OBJECT1_CDRG_N',
        't_cdrg_rdot':'OBJECT1_CDRG_RDOT',
        't_cdrg_tdot':'OBJECT1_CDRG_TDOT',
        't_cdrg_ndot':'OBJECT1_CDRG_NDOT',
        't_cdrg_drg':'OBJECT1_CDRG_DRG',
        't_csrp_r':'OBJECT1_CSRP_R',
        't_csrp_t':'OBJECT1_CSRP_T',
        't_csrp_n':'OBJECT1_CSRP_N',
        't_csrp_rdot':'OBJECT1_CSRP_RDOT',
        't_csrp_tdot':'OBJECT1_CSRP_TDOT',
        't_csrp_ndot':'OBJECT1_CSRP_NDOT',
        't_csrp_drg':'OBJECT1_CSRP_DRG',
        't_csrp_srp':'OBJECT1_CSRP_SRP',
        't_cthr_r':'OBJECT1_CTHR_R',
        't_cthr_t':'OBJECT1_CTHR_T',
        't_cthr_n':'OBJECT1_CTHR_N',
        't_cthr_rdot':'OBJECT1_CTHR_RDOT',
        't_cthr_tdot':'OBJECT1_CTHR_TDOT',
        't_cthr_ndot':'OBJECT1_CTHR_NDOT',
        't_cthr_drg':'OBJECT1_CTHR_DRG',
        't_cthr_srp':'OBJECT1_CTHR_SRP',
        't_cthr_thr':'OBJECT1_CTHR_THR',
        'c_object_designator':'OBJECT2_OBJECT_DESIGNATOR',
        'c_catalog_name':'OBJECT2_CATALOG_NAME',
        'c_object_name':'OBJECT2_OBJECT_NAME',
        'c_international_designator':'OBJECT2_INTERNATIONAL_DESIGNATOR',
        'c_object_type':'OBJECT2_OBJECT_TYPE',
        'c_ephemeris_name':'OBJECT2_EPHEMERIS_NAME',
        'c_covariance_method':'OBJECT2_COVARIANCE_METHOD',
        'c_maneuverable':'OBJECT2_MANEUVERABLE',
        'c_orbit_center':'OBJECT2_ORBIT_CENTER',
        'c_ref_frame':'OBJECT2_REF_FRAME',
        'c_gravity_model':'OBJECT2_GRAVITY_MODEL',
        'c_atmospheric_model':'OBJECT2_ATMOSPHERIC_MODEL',
        'c_n_body_perturbations':'OBJECT2_N_BODY_PERTURBATIONS',
        'c_solar_rad_pressure':'OBJECT2_SOLAR_RAD_PRESSURE',
        'c_earth_tides':'OBJECT2_EARTH_TIDES',
        'c_intrack_thrust':'OBJECT2_INTRACK_THRUST',
        'c_time_lastob_start':'OBJECT2_TIME_LASTOB_START',
        'c_time_lastob_end':'OBJECT2_TIME_LASTOB_END',
        'c_recommended_od_span':'OBJECT2_RECOMMENDED_OD_SPAN',
        'c_actual_od_span':'OBJECT2_ACTUAL_OD_SPAN',
        'c_obs_available':'OBJECT2_OBS_AVAILABLE',
        'c_obs_used':'OBJECT2_OBS_USED',
        'c_tracks_available':'OBJECT2_TRACKS_AVAILABLE',
        'c_tracks_used':'OBJECT2_TRACKS_USED',
        'c_residuals_accepted':'OBJECT2_RESIDUALS_ACCEPTED',
        'c_weighted_rms':'OBJECT2_WEIGHTED_RMS',
        'c_area_pc':'OBJECT2_AREA_PC',
        'c_area_drg':'OBJECT2_AREA_DRG',
        'c_area_srg':'OBJECT2_AREA_SRP',
        'c_mass':'OBJECT2_MASS',
        'c_cd_area_over_mass':'OBJECT2_CD_AREA_OVER_MASS',
        'c_cr_area_over_mass':'OBJECT2_CR_AREA_OVER_MASS',
        'c_thrust_acceleration':'OBJECT2_THRUST_ACCELERATION',
        'c_sedr':'OBJECT2_SEDR',
        'c_x':'OBJECT2_X',
        'c_y':'OBJECT2_Y',
        'c_z':'OBJECT2_Z',
        'c_x_dot':'OBJECT2_X_DOT',
        'c_y_dot':'OBJECT2_Y_DOT',
        'c_z_dot':'OBJECT2_Z_DOT',
        'c_cr_r':'OBJECT2_CR_R',
        'c_ct_r':'OBJECT2_CT_R',
        'c_ct_t':'OBJECT2_CT_T',
        'c_cn_r':'OBJECT2_CN_R',
        'c_cn_t':'OBJECT2_CN_T',
        'c_cn_n':'OBJECT2_CN_N',
        'c_crdot_r':'OBJECT2_CRDOT_R',
        'c_crdot_t':'OBJECT2_CRDOT_T',
        'c_crdot_n':'OBJECT2_CRDOT_N',
        'c_crdot_rdot':'OBJECT2_CRDOT_RDOT',
        'c_ctdot_r':'OBJECT2_CTDOT_R',
        'c_ctdot_t':'OBJECT2_CTDOT_T',
        'c_ctdot_n':'OBJECT2_CTDOT_N',
        'c_ctdot_rdot':'OBJECT2_CTDOT_RDOT',
        'c_ctdot_tdot':'OBJECT2_CTDOT_TDOT',
        'c_cndot_r':'OBJECT2_CNDOT_R',
        'c_cndot_t':'OBJECT2_CNDOT_T',
        'c_cndot_n':'OBJECT2_CNDOT_N',
        'c_cndot_rdot':'OBJECT2_CNDOT_RDOT',
        'c_cndot_tdot':'OBJECT2_CNDOT_TDOT',
        'c_cndot_ndot':'OBJECT2_CNDOT_NDOT',
        'c_cdrg_r':'OBJECT2_CDRG_R',
        'c_cdrg_t':'OBJECT2_CDRG_T',
        'c_cdrg_n':'OBJECT2_CDRG_N',
        'c_cdrg_rdot':'OBJECT2_CDRG_RDOT',
        'c_cdrg_tdot':'OBJECT2_CDRG_TDOT',
        'c_cdrg_ndot':'OBJECT2_CDRG_NDOT',
        'c_cdrg_drg':'OBJECT2_CDRG_DRG',
        'c_csrp_r':'OBJECT2_CSRP_R',
        'c_csrp_t':'OBJECT2_CSRP_T',
        'c_csrp_n':'OBJECT2_CSRP_N',
        'c_csrp_rdot':'OBJECT2_CSRP_RDOT',
        'c_csrp_tdot':'OBJECT2_CSRP_TDOT',
        'c_csrp_ndot':'OBJECT2_CSRP_NDOT',
        'c_csrp_drg':'OBJECT2_CSRP_DRG',
        'c_csrp_srp':'OBJECT2_CSRP_SRP',
        'c_cthr_r':'OBJECT2_CTHR_R',
        'c_cthr_t':'OBJECT2_CTHR_T',
        'c_cthr_n':'OBJECT2_CTHR_N',
        'c_cthr_rdot':'OBJECT2_CTHR_RDOT',
        'c_cthr_tdot':'OBJECT2_CTHR_TDOT',
        'c_cthr_ndot':'OBJECT2_CTHR_NDOT',
        'c_cthr_drg':'OBJECT2_CTHR_DRG',
        'c_cthr_srp':'OBJECT2_CTHR_SRP',
        'c_cthr_thr':'OBJECT2_CTHR_THR'}, group_events_by='event_id', date_format='%Y-%m-%d %H:%M:%S.%f'):

        print('Dataframe with {} rows and {} columns'.format(len(df), len(df.columns)))
        print('Dropping columns with NaNs')
        df = df.dropna(axis=1)
        print('Dataframe with {} rows and {} columns'.format(len(df), len(df.columns)))
        pandas_column_names_after_dropping = list(df.columns)

        print('Grouping by {}'.format(group_events_by))
        df_events = df.groupby(group_events_by).groups
        print('Grouped into {} event(s)'.format(len(df_events)))
        events = []
        util.progress_bar_init('Converting DataFrame to EventDataset', len(df_events), 'Events')
        i = 0
        for k, v in df_events.items():
            util.progress_bar_update(i)
            i += 1
            df_event = df.iloc[v]
            cdms = []
            for _, df_cdm in df_event.iterrows():
                cdm = CDM()
                for pandas_name, cdm_name in cdm_compatible_fields.items():
                    if pandas_name in pandas_column_names_after_dropping:
                        value = df_cdm[pandas_name]
                        # Check if the field is a date, if so transform to the correct date string format expected in the CCSDS 508.0-B-1 standard
                        if util.is_date(value, date_format):
                            value = util.transform_date_str(value, date_format, '%Y-%m-%dT%H:%M:%S.%f')
                        cdm[cdm_name] = value
                cdms.append(cdm)
            events.append(Event(cdms))
        util.progress_bar_end()
        event_dataset = EventDataset(events=events)
        print('\n{}'.format(event_dataset))
        return event_dataset

    def to_dataframe(self):
        if len(self) == 0:
            return pd.DataFrame()
        event_dataframes = []

        util.progress_bar_init('Converting EventDataset to DataFrame', len(self._events), 'Events')
        for i, event in enumerate(self._events):
            util.progress_bar_update(i)
            event_dataframes.append(event.to_dataframe())
        util.progress_bar_end()
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

    def common_features(self, only_numeric=False):
        df = self.to_dataframe()
        df = df.dropna(axis=1)
        if only_numeric:
            df = df.select_dtypes(include=['int', 'float64', 'float32'])
        features = list(df.columns)
        if '__DAYS_TO_TCA' in features:
            features.remove('__DAYS_TO_TCA')
        return features

    def get_CDMs(self):
        cdms = []
        for event in self:
            for cdm in event:
                cdms.append(cdm)
        return cdms

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
            features = ['CR_R','CT_T', 'CN_N', 'CRDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_NDOT']
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
