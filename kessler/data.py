import uuid
import os
import torch
import pandas as pd
import sys
from datetime import datetime, timedelta

from . import util
from .models import Conjunction
from .cdm import CDM
from .event import Event, EventDataset


def generate_event_dataset(dataset_dir, num_events, save_traces=False, *args, **kwargs,verbosity=True):
    model = Conjunction(*args, **kwargs)
    if verbosity:
        print('Generating CDM dataset')
        print('Directory: {}'.format(dataset_dir))

    util.create_path(dataset_dir, directory=True)
    for i in range(num_events):
        if verbosity:
            print('Generating event {} / {}'.format(i+1, num_events))
        file_name_event = os.path.join(dataset_dir, 'event_{}'.format(str(uuid.uuid4())))

        trace = model.get_conjunction()
        if save_traces:
            file_name_trace = file_name_event + '.trace'
            if verbosity:
                print('Saving trace: {}'.format(file_name_trace))
            torch.save(trace, file_name_trace)

        cdms = trace['cdms']
        for j, cdm in enumerate(cdms):
            file_name_suffix = '{}'.format(j).rjust(len('{}'.format(len(cdms))), '0')
            file_name_cdm = file_name_event + '_{}.cdm.kvn.txt'.format(file_name_suffix)
            if verbosity:
                print('Saving cdm  : {}'.format(file_name_cdm))
            cdm.save(file_name_cdm)



def kelvins_to_event_dataset(file_name, num_events=None, date_tca=None, remove_outliers=True, drop_features=['c_rcs_estimate', 't_rcs_estimate']):
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
            cdm = CDM()
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
            cdm['OBJECT1_CT_R'] = kelvins_cdm['t_ct_r'] * kelvins_cdm['t_sigma_r'] * kelvins_cdm['t_sigma_t']
            cdm['OBJECT1_CT_T'] = kelvins_cdm['t_sigma_t']**2.
            cdm['OBJECT1_CN_R'] = kelvins_cdm['t_cn_r'] * kelvins_cdm['t_sigma_n'] * kelvins_cdm['t_sigma_r']
            cdm['OBJECT1_CN_T'] = kelvins_cdm['t_cn_t'] * kelvins_cdm['t_sigma_n'] * kelvins_cdm['t_sigma_t']
            cdm['OBJECT1_CN_N'] = kelvins_cdm['t_sigma_n']**2.
            cdm['OBJECT1_CRDOT_R'] = kelvins_cdm['t_crdot_r'] * kelvins_cdm['t_sigma_rdot'] * kelvins_cdm['t_sigma_r']
            cdm['OBJECT1_CRDOT_T'] = kelvins_cdm['t_crdot_t'] * kelvins_cdm['t_sigma_rdot'] * kelvins_cdm['t_sigma_t']
            cdm['OBJECT1_CRDOT_N'] = kelvins_cdm['t_crdot_n'] * kelvins_cdm['t_sigma_rdot'] * kelvins_cdm['t_sigma_n']
            cdm['OBJECT1_CRDOT_RDOT'] = kelvins_cdm['t_sigma_rdot']**2.
            cdm['OBJECT1_CTDOT_R'] = kelvins_cdm['t_ctdot_r'] * kelvins_cdm['t_sigma_tdot'] * kelvins_cdm['t_sigma_r']
            cdm['OBJECT1_CTDOT_T'] = kelvins_cdm['t_ctdot_t'] * kelvins_cdm['t_sigma_tdot'] * kelvins_cdm['t_sigma_t']
            cdm['OBJECT1_CTDOT_N'] = kelvins_cdm['t_ctdot_n'] * kelvins_cdm['t_sigma_tdot'] * kelvins_cdm['t_sigma_n']
            cdm['OBJECT1_CTDOT_RDOT'] = kelvins_cdm['t_ctdot_rdot'] * kelvins_cdm['t_sigma_tdot'] * kelvins_cdm['t_sigma_rdot']
            cdm['OBJECT1_CTDOT_TDOT'] = kelvins_cdm['t_sigma_tdot']**2.
            cdm['OBJECT1_CNDOT_R'] = kelvins_cdm['t_cndot_r'] * kelvins_cdm['t_sigma_ndot'] * kelvins_cdm['t_sigma_r']
            cdm['OBJECT1_CNDOT_T'] = kelvins_cdm['t_cndot_t'] * kelvins_cdm['t_sigma_ndot'] * kelvins_cdm['t_sigma_t']
            cdm['OBJECT1_CNDOT_N'] = kelvins_cdm['t_cndot_n'] * kelvins_cdm['t_sigma_ndot'] * kelvins_cdm['t_sigma_n']
            cdm['OBJECT1_CNDOT_RDOT'] = kelvins_cdm['t_cndot_rdot'] * kelvins_cdm['t_sigma_ndot'] * kelvins_cdm['t_sigma_rdot']
            cdm['OBJECT1_CNDOT_TDOT'] = kelvins_cdm['t_cndot_tdot'] * kelvins_cdm['t_sigma_ndot'] * kelvins_cdm['t_sigma_tdot']
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
            cdm['OBJECT2_CT_R'] = kelvins_cdm['c_ct_r'] * kelvins_cdm['c_sigma_r'] * kelvins_cdm['c_sigma_t']
            cdm['OBJECT2_CT_T'] = kelvins_cdm['c_sigma_t']**2.
            cdm['OBJECT2_CN_R'] = kelvins_cdm['c_cn_r'] * kelvins_cdm['c_sigma_n'] * kelvins_cdm['c_sigma_r']
            cdm['OBJECT2_CN_T'] = kelvins_cdm['c_cn_t'] * kelvins_cdm['c_sigma_n'] * kelvins_cdm['c_sigma_t']
            cdm['OBJECT2_CN_N'] = kelvins_cdm['c_sigma_n']**2.
            cdm['OBJECT2_CRDOT_R'] = kelvins_cdm['c_crdot_r'] * kelvins_cdm['c_sigma_rdot'] * kelvins_cdm['c_sigma_r']
            cdm['OBJECT2_CRDOT_T'] = kelvins_cdm['c_crdot_t'] * kelvins_cdm['c_sigma_rdot'] * kelvins_cdm['c_sigma_t']
            cdm['OBJECT2_CRDOT_N'] = kelvins_cdm['c_crdot_n'] * kelvins_cdm['c_sigma_rdot'] * kelvins_cdm['c_sigma_n']
            cdm['OBJECT2_CRDOT_RDOT'] = kelvins_cdm['c_sigma_rdot']**2.
            cdm['OBJECT2_CTDOT_R'] = kelvins_cdm['c_ctdot_r'] * kelvins_cdm['c_sigma_tdot'] * kelvins_cdm['c_sigma_r']
            cdm['OBJECT2_CTDOT_T'] = kelvins_cdm['c_ctdot_t'] * kelvins_cdm['c_sigma_tdot'] * kelvins_cdm['c_sigma_t']
            cdm['OBJECT2_CTDOT_N'] = kelvins_cdm['c_ctdot_n'] * kelvins_cdm['c_sigma_tdot'] * kelvins_cdm['c_sigma_n']
            cdm['OBJECT2_CTDOT_RDOT'] = kelvins_cdm['c_ctdot_rdot'] * kelvins_cdm['c_sigma_tdot'] * kelvins_cdm['c_sigma_rdot']
            cdm['OBJECT2_CTDOT_TDOT'] = kelvins_cdm['c_sigma_tdot']**2.
            cdm['OBJECT2_CNDOT_R'] = kelvins_cdm['c_cndot_r'] * kelvins_cdm['c_sigma_ndot'] * kelvins_cdm['c_sigma_r']
            cdm['OBJECT2_CNDOT_T'] = kelvins_cdm['c_cndot_t'] * kelvins_cdm['c_sigma_ndot'] * kelvins_cdm['c_sigma_t']
            cdm['OBJECT2_CNDOT_N'] = kelvins_cdm['c_cndot_n'] * kelvins_cdm['c_sigma_ndot'] * kelvins_cdm['c_sigma_n']
            cdm['OBJECT2_CNDOT_RDOT'] = kelvins_cdm['c_cndot_rdot'] * kelvins_cdm['c_sigma_ndot'] * kelvins_cdm['c_sigma_rdot']
            cdm['OBJECT2_CNDOT_TDOT'] = kelvins_cdm['c_cndot_tdot'] * kelvins_cdm['c_sigma_ndot'] * kelvins_cdm['c_sigma_tdot']
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

    return EventDataset(events=events)
