# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
# Trillium Technologies
# Giacomo Acciarini
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.


import numpy as np
import warnings
import datetime
import copy
import pandas as pd
from . import util


# Based on CCSDS 508.0-B-1
# https://public.ccsds.org/Pubs/508x0b1e2c1.pdf
class ConjunctionDataMessage():
    def __init__(self, file_name=None, set_defaults=True):
        # Header
        # Relative metadata
        # Object 1
        #  Metadata, OD, State, Covariance
        # Object 2
        #  Metadata, OD, State, Covariance
        # Comments are optional and not currently supported by this class

        self._keys_header = ['CCSDS_CDM_VERS', 'CREATION_DATE', 'ORIGINATOR', 'MESSAGE_FOR', 'MESSAGE_ID']
        self._keys_relative_metadata = ['TCA', 'MISS_DISTANCE', 'RELATIVE_SPEED', 'RELATIVE_POSITION_R', 'RELATIVE_POSITION_T', 'RELATIVE_POSITION_N', 'RELATIVE_VELOCITY_R', 'RELATIVE_VELOCITY_T', 'RELATIVE_VELOCITY_N', 'START_SCREEN_PERIOD', 'STOP_SCREEN_PERIOD', 'SCREEN_VOLUME_FRAME', 'SCREEN_VOLUME_SHAPE', 'SCREEN_VOLUME_X', 'SCREEN_VOLUME_Y', 'SCREEN_VOLUME_Z', 'SCREEN_ENTRY_TIME', 'SCREEN_EXIT_TIME', 'COLLISION_PROBABILITY', 'COLLISION_PROBABILITY_METHOD']
        self._keys_metadata = ['OBJECT', 'OBJECT_DESIGNATOR', 'CATALOG_NAME', 'OBJECT_NAME', 'INTERNATIONAL_DESIGNATOR', 'OBJECT_TYPE', 'OPERATOR_CONTACT_POSITION', 'OPERATOR_ORGANIZATION', 'OPERATOR_PHONE', 'OPERATOR_EMAIL', 'EPHEMERIS_NAME', 'COVARIANCE_METHOD', 'MANEUVERABLE', 'ORBIT_CENTER', 'REF_FRAME', 'GRAVITY_MODEL', 'ATMOSPHERIC_MODEL', 'N_BODY_PERTURBATIONS', 'SOLAR_RAD_PRESSURE', 'EARTH_TIDES', 'INTRACK_THRUST']
        self._keys_data_od = ['TIME_LASTOB_START', 'TIME_LASTOB_END', 'RECOMMENDED_OD_SPAN', 'ACTUAL_OD_SPAN', 'OBS_AVAILABLE', 'OBS_USED', 'TRACKS_AVAILABLE', 'TRACKS_USED', 'RESIDUALS_ACCEPTED', 'WEIGHTED_RMS', 'AREA_PC', 'AREA_DRG', 'AREA_SRP', 'MASS', 'CD_AREA_OVER_MASS', 'CR_AREA_OVER_MASS', 'THRUST_ACCELERATION', 'SEDR']
        self._keys_data_state = ['X', 'Y', 'Z', 'X_DOT', 'Y_DOT', 'Z_DOT']
        self._keys_data_covariance = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT', 'CDRG_R', 'CDRG_T', 'CDRG_N', 'CDRG_RDOT', 'CDRG_TDOT', 'CDRG_NDOT', 'CDRG_DRG', 'CSRP_R', 'CSRP_T', 'CSRP_N', 'CSRP_RDOT', 'CSRP_TDOT', 'CSRP_NDOT', 'CSRP_DRG', 'CSRP_SRP', 'CTHR_R', 'CTHR_T', 'CTHR_N', 'CTHR_RDOT', 'CTHR_TDOT', 'CTHR_NDOT', 'CTHR_DRG', 'CTHR_SRP', 'CTHR_THR']

        self._keys_header_obligatory = ['CCSDS_CDM_VERS', 'CREATION_DATE', 'ORIGINATOR', 'MESSAGE_ID']
        self._keys_relative_metadata_obligatory = ['TCA', 'MISS_DISTANCE']
        self._keys_metadata_obligatory = ['OBJECT', 'OBJECT_DESIGNATOR', 'CATALOG_NAME', 'OBJECT_NAME', 'INTERNATIONAL_DESIGNATOR', 'EPHEMERIS_NAME', 'COVARIANCE_METHOD', 'MANEUVERABLE', 'REF_FRAME']
        self._keys_data_od_obligatory = []
        self._keys_data_state_obligatory = ['X', 'Y', 'Z', 'X_DOT', 'Y_DOT', 'Z_DOT']
        self._keys_data_covariance_obligatory = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']

        self._values_header = dict.fromkeys(self._keys_header)
        self._values_relative_metadata = dict.fromkeys(self._keys_relative_metadata)
        self._values_object_metadata = [dict.fromkeys(self._keys_metadata), dict.fromkeys(self._keys_metadata)]
        self._values_object_data_od = [dict.fromkeys(self._keys_data_od), dict.fromkeys(self._keys_data_od)]
        self._values_object_data_state = [dict.fromkeys(self._keys_data_state), dict.fromkeys(self._keys_data_state)]
        self._values_object_data_covariance = [dict.fromkeys(self._keys_data_covariance), dict.fromkeys(self._keys_data_covariance)]
        self._values_extra = {}  # This holds extra key, value pairs associated with each CDM object, used internally by the Kessler codebase and not a part of the CDM standard

        self._keys_with_dates = ['CREATION_DATE', 'TCA', 'SCREEN_ENTRY_TIME', 'START_SCREEN_PERIOD', 'STOP_SCREEN_PERIOD', 'SCREEN_EXIT_TIME', 'OBJECT1_TIME_LASTOB_START', 'OBJECT1_TIME_LASTOB_END', 'OBJECT2_TIME_LASTOB_START', 'OBJECT2_TIME_LASTOB_END']

        if set_defaults:
            self.set_header('CCSDS_CDM_VERS', '1.0')
            self.set_header('CREATION_DATE', datetime.datetime.utcnow().isoformat())
            self.set_object(0, 'OBJECT', 'OBJECT1')
            self.set_object(1, 'OBJECT', 'OBJECT2')

        if file_name:
            self.copy_from(ConjunctionDataMessage.load(file_name))

    def copy(self):
        ret = ConjunctionDataMessage()
        ret._values_header = copy.deepcopy(self._values_header)
        ret._values_relative_metadata = copy.deepcopy(self._values_relative_metadata)
        ret._values_object_metadata = copy.deepcopy(self._values_object_metadata)
        ret._values_object_data_od = copy.deepcopy(self._values_object_data_od)
        ret._values_object_data_state = copy.deepcopy(self._values_object_data_state)
        ret._values_object_data_covariance = copy.deepcopy(self._values_object_data_covariance)
        return ret

    def copy_from(self, other_cdm):
        self._values_header = copy.deepcopy(other_cdm._values_header)
        self._values_relative_metadata = copy.deepcopy(other_cdm._values_relative_metadata)
        self._values_object_metadata = copy.deepcopy(other_cdm._values_object_metadata)
        self._values_object_data_od = copy.deepcopy(other_cdm._values_object_data_od)
        self._values_object_data_state = copy.deepcopy(other_cdm._values_object_data_state)
        self._values_object_data_covariance = copy.deepcopy(other_cdm._values_object_data_covariance)

    def to_dict(self):
        data = {}
        data_header = dict.fromkeys(self._keys_header)
        for key, value in self._values_header.items():
            data_header[key] = value
        data.update(data_header)

        data_relative_metadata = dict.fromkeys(self._keys_relative_metadata)
        for key, value in self._values_relative_metadata.items():
            data_relative_metadata[key] = value
        data.update(data_relative_metadata)

        for i in [0, 1]:
            prefix = 'OBJECT{}_'.format(i+1)
            keys_metadata = map(lambda x: prefix+x, self._keys_metadata)
            keys_data_od = map(lambda x: prefix+x, self._keys_data_od)
            keys_data_state = map(lambda x: prefix+x, self._keys_data_state)
            keys_data_covariance = map(lambda x: prefix+x, self._keys_data_covariance)

            data_metadata = dict.fromkeys(keys_metadata)
            for key, value in self._values_object_metadata[i].items():
                data_metadata[prefix+key] = value
            data.update(data_metadata)

            data_data_od = dict.fromkeys(keys_data_od)
            for key, value in self._values_object_data_od[i].items():
                data_data_od[prefix+key] = value
            data.update(data_data_od)

            data_data_state = dict.fromkeys(keys_data_state)
            for key, value in self._values_object_data_state[i].items():
                data_data_state[prefix+key] = value
            data.update(data_data_state)

            data_data_covariance = dict.fromkeys(keys_data_covariance)
            for key, value in self._values_object_data_covariance[i].items():
                data_data_covariance[prefix+key] = value
            data.update(data_data_covariance)

        data.update(self._values_extra)

        return data

    def to_dataframe(self):
        data = self.to_dict()
        return pd.DataFrame(data, index=[0])

    def load(file_name):
        content = []
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace(u'\ufeff', '')
                line = line.strip()
                if line.startswith('COMMENT') or len(line) == 0:
                    continue
                key, value = line.split('=')
                key, value = key.strip(), value.strip()
                if util.is_number(value):
                    value = float(value)
    #             print(line)
                content.append((key, value))
        cdm = ConjunctionDataMessage(set_defaults=False)
        currently_parsing = 'header_and_relative_metadata'
        for key, value in content:
            if currently_parsing == 'header_and_relative_metadata':
                if key in cdm._keys_header:
                    cdm.set_header(key, value)
                elif key in cdm._keys_relative_metadata:
                    cdm.set_relative_metadata(key, value)
                elif key == 'OBJECT' and value == 'OBJECT1':
                    cdm.set_object(0, key, value)
                    currently_parsing = 'object1'
                    continue
                elif key == 'OBJECT' and value == 'OBJECT2':
                    cdm.set_object(1, key, value)
                    currently_parsing = 'object2'
                    continue
            elif currently_parsing == 'object1':
                if key == 'OBJECT' and value == 'OBJECT2':
                    cdm.set_object(1, key, value)
                    currently_parsing = 'object2'
                    continue
                try:
                    cdm.set_object(0, key, value)
                except:
                    continue
            elif currently_parsing == 'object2':
                if key == 'OBJECT' and value == 'OBJECT1':
                    cdm.set_object(0, key, value)
                    currently_parsing = 'object1'
                    continue
                try:
                    cdm.set_object(1, key, value)
                except:
                    continue
        return cdm

    def save(self, file_name):
        content = self.kvn()
        with open(file_name, 'w') as f:
            f.write(content)

    def __hash__(self):
        return hash(self.kvn(show_all=True))

    def __eq__(self, other):
        if isinstance(other, ConjunctionDataMessage):
            return hash(self) == hash(other)
        return False
    
    def set_header(self, key, value):
        if key in self._keys_header:
            if key in self._keys_with_dates:
                # We have a field with a date string as the value. Check if the string is in the format needed by the CCSDS 508.0-B-1 standard
                time_format = util.get_ccsds_time_format(value)
                idx = time_format.find('DDD')
                if idx!=-1:
                    value = util.doy_2_date(value, value[idx:idx+3], value[0:4], idx)
                try:
                    _ = datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')
                except Exception as e:
                    raise RuntimeError('{} ({}) is not in the expected format.\n{}'.format(key, value, str(e)))
            self._values_header[key] = value
        else:
            raise ValueError('Invalid key ({}) for header'.format(key))

    def set_relative_metadata(self, key, value):
        if key in self._keys_relative_metadata:
            self._values_relative_metadata[key] = value
        else:
            raise ValueError('Invalid key ({}) for relative metadata'.format(key))

    def set_object(self, object_id, key, value):
        if object_id != 0 and object_id != 1:
            raise ValueError('Expecting object_id to be 0 or 1')
        if key in self._keys_metadata:
            self._values_object_metadata[object_id][key] = value
        elif key in self._keys_data_od:
            self._values_object_data_od[object_id][key] = value
        elif key in self._keys_data_state:
            self._values_object_data_state[object_id][key] = value
        elif key in self._keys_data_covariance:
            self._values_object_data_covariance[object_id][key] = value
        else:
            raise ValueError('Invalid key ({}) for object data'.format(key))

    def get_object(self, object_id, key):
        if object_id != 0 and object_id != 1:
            raise ValueError('Expecting object_id to be 0 or 1')
        if key in self._keys_metadata:
            return self._values_object_metadata[object_id][key]
        elif key in self._keys_data_od:
            return self._values_object_data_od[object_id][key]
        elif key in self._keys_data_state:
            return self._values_object_data_state[object_id][key]
        elif key in self._keys_data_covariance:
            return self._values_object_data_covariance[object_id][key]
        else:
            raise ValueError('Invalid key ({}) for object data'.format(key))

    def get_relative_metadata(self, key):
        if key in self._keys_relative_metadata:
            return self._values_relative_metadata[key]
        else:
            raise ValueError('Invalid key ({}) for relative metadata'.format(key))

    def set_state(self, object_id, state):
        self.set_object(object_id, 'X', state[0, 0])
        self.set_object(object_id, 'Y', state[0, 1])
        self.set_object(object_id, 'Z', state[0, 2])
        self.set_object(object_id, 'X_DOT', state[1, 0])
        self.set_object(object_id, 'Y_DOT', state[1, 1])
        self.set_object(object_id, 'Z_DOT', state[1, 2])
        self._update_state_relative()
        self._update_miss_distance()

    def _update_miss_distance(self):
        state_object1 = self.get_state(0)
        if np.isnan(state_object1.sum()):
            warnings.warn('state_object1 has NaN')
        state_object2 = self.get_state(1)
        if np.isnan(state_object2.sum()):
            warnings.warn('state_object2 has NaN')

        miss_distance = np.linalg.norm(state_object1[0] - state_object2[0])
        self.set_relative_metadata('MISS_DISTANCE', miss_distance)

    def _update_state_relative(self):
        def uvw_matrix(r, v):
            u = r / np.linalg.norm(r)
            w = np.cross(r, v)
            w = w / np.linalg.norm(w)
            v = np.cross(w, u)
            return np.vstack((u, v, w))

        # Takes states in ITRF and returns relative state in RTN with target as reference
        def relative_state(state_obj_1, state_obj_2):
            rot_matrix = uvw_matrix(state_obj_1[0], state_obj_1[1])
            rel_position_xyz = state_obj_2[0] - state_obj_1[0]
            rel_velocity_xyz = state_obj_2[1] - state_obj_1[1]
            relative_state = np.zeros([2, 3])
            relative_state[0] = np.array([np.dot(rot_matrix[0], rel_position_xyz), np.dot(rot_matrix[1], rel_position_xyz), np.dot(rot_matrix[2], rel_position_xyz)])
            relative_state[1] = np.array([np.dot(rot_matrix[0], rel_velocity_xyz), np.dot(rot_matrix[1], rel_velocity_xyz), np.dot(rot_matrix[2], rel_velocity_xyz)])
            return relative_state

        state_object1 = self.get_state(0)
        if np.isnan(state_object1.sum()):
            warnings.warn('state_object1 has NaN')
        state_object2 = self.get_state(1)
        if np.isnan(state_object2.sum()):
            warnings.warn('state_object2 has NaN')

        relative_state = relative_state(state_object1, state_object2)

        self.set_relative_metadata('RELATIVE_POSITION_R', relative_state[0, 0])
        self.set_relative_metadata('RELATIVE_POSITION_T', relative_state[0, 1])
        self.set_relative_metadata('RELATIVE_POSITION_N', relative_state[0, 2])
        self.set_relative_metadata('RELATIVE_VELOCITY_R', relative_state[1, 0])
        self.set_relative_metadata('RELATIVE_VELOCITY_T', relative_state[1, 1])
        self.set_relative_metadata('RELATIVE_VELOCITY_N', relative_state[1, 2])
        self.set_relative_metadata('RELATIVE_SPEED', np.linalg.norm(relative_state[1]))

    def get_state_relative(self):
        relative_state = np.zeros([2, 3])
        relative_state[0, 0] = self.get_relative_metadata('RELATIVE_POSITION_R')
        relative_state[0, 1] = self.get_relative_metadata('RELATIVE_POSITION_T')
        relative_state[0, 2] = self.get_relative_metadata('RELATIVE_POSITION_N')
        relative_state[1, 0] = self.get_relative_metadata('RELATIVE_VELOCITY_R')
        relative_state[1, 1] = self.get_relative_metadata('RELATIVE_VELOCITY_T')
        relative_state[1, 2] = self.get_relative_metadata('RELATIVE_VELOCITY_N')
        return relative_state

    def get_state(self, object_id):
        state = np.zeros([2, 3])
        state[0, 0] = self.get_object(object_id, 'X')
        state[0, 1] = self.get_object(object_id, 'Y')
        state[0, 2] = self.get_object(object_id, 'Z')
        state[1, 0] = self.get_object(object_id, 'X_DOT')
        state[1, 1] = self.get_object(object_id, 'Y_DOT')
        state[1, 2] = self.get_object(object_id, 'Z_DOT')
        return state

    def get_covariance(self, object_id):
        covariance = np.zeros([6, 6])
        covariance[0, 0] = self.get_object(object_id, 'CR_R')
        covariance[1, 0] = self.get_object(object_id, 'CT_R')
        covariance[1, 1] = self.get_object(object_id, 'CT_T')
        covariance[2, 0] = self.get_object(object_id, 'CN_R')
        covariance[2, 1] = self.get_object(object_id, 'CN_T')
        covariance[2, 2] = self.get_object(object_id, 'CN_N')
        covariance[3, 0] = self.get_object(object_id, 'CRDOT_R')
        covariance[3, 1] = self.get_object(object_id, 'CRDOT_T')
        covariance[3, 2] = self.get_object(object_id, 'CRDOT_N')
        covariance[3, 3] = self.get_object(object_id, 'CRDOT_RDOT')
        covariance[4, 0] = self.get_object(object_id, 'CTDOT_R')
        covariance[4, 1] = self.get_object(object_id, 'CTDOT_T')
        covariance[4, 2] = self.get_object(object_id, 'CTDOT_N')
        covariance[4, 3] = self.get_object(object_id, 'CTDOT_RDOT')
        covariance[4, 4] = self.get_object(object_id, 'CTDOT_TDOT')
        covariance[5, 0] = self.get_object(object_id, 'CNDOT_R')
        covariance[5, 1] = self.get_object(object_id, 'CNDOT_T')
        covariance[5, 2] = self.get_object(object_id, 'CNDOT_N')
        covariance[5, 3] = self.get_object(object_id, 'CNDOT_RDOT')
        covariance[5, 4] = self.get_object(object_id, 'CNDOT_TDOT')
        covariance[5, 5] = self.get_object(object_id, 'CNDOT_NDOT')
        # Copies lower triangle to the upper part
        covariance = covariance + covariance.T - np.diag(np.diag(covariance))
        return covariance

    def set_covariance(self, object_id, covariance_matrix):
        self.set_object(object_id, 'CR_R', covariance_matrix[0, 0])
        self.set_object(object_id, 'CT_R', covariance_matrix[1, 0])
        self.set_object(object_id, 'CT_T', covariance_matrix[1, 1])
        self.set_object(object_id, 'CN_R', covariance_matrix[2, 0])
        self.set_object(object_id, 'CN_T', covariance_matrix[2, 1])
        self.set_object(object_id, 'CN_N', covariance_matrix[2, 2])
        self.set_object(object_id, 'CRDOT_R', covariance_matrix[3, 0])
        self.set_object(object_id, 'CRDOT_T', covariance_matrix[3, 1])
        self.set_object(object_id, 'CRDOT_N', covariance_matrix[3, 2])
        self.set_object(object_id, 'CRDOT_RDOT', covariance_matrix[3, 3])
        self.set_object(object_id, 'CTDOT_R', covariance_matrix[4, 0])
        self.set_object(object_id, 'CTDOT_T', covariance_matrix[4, 1])
        self.set_object(object_id, 'CTDOT_N', covariance_matrix[4, 2])
        self.set_object(object_id, 'CTDOT_RDOT', covariance_matrix[4, 3])
        self.set_object(object_id, 'CTDOT_TDOT', covariance_matrix[4, 4])
        self.set_object(object_id, 'CNDOT_R', covariance_matrix[5, 0])
        self.set_object(object_id, 'CNDOT_T', covariance_matrix[5, 1])
        self.set_object(object_id, 'CNDOT_N', covariance_matrix[5, 2])
        self.set_object(object_id, 'CNDOT_RDOT', covariance_matrix[5, 3])
        self.set_object(object_id, 'CNDOT_TDOT', covariance_matrix[5, 4])
        self.set_object(object_id, 'CNDOT_NDOT', covariance_matrix[5, 5])

    def validate(self):
        def val(keys, vals, part):
            for key in keys:
                if vals[key] is None:
                    print('Missing obligatory value in {}: {}'.format(part, key))
        val(self._keys_header_obligatory, self._values_header, 'header')
        val(self._keys_relative_metadata_obligatory, self._values_relative_metadata, 'relative metadata')
        val(self._keys_metadata_obligatory, self._values_object_metadata[0], 'Object1 metadata')
        val(self._keys_data_od_obligatory, self._values_object_data_od[0], 'Object1 data (od)')
        val(self._keys_data_state_obligatory, self._values_object_data_state[0], 'Object1 data (state)')
        val(self._keys_data_covariance_obligatory, self._values_object_data_covariance[0], 'Object1 data (covariance)')
        val(self._keys_metadata_obligatory, self._values_object_metadata[1], 'Object2 metadata')
        val(self._keys_data_od_obligatory, self._values_object_data_od[1], 'Object2 data (od)')
        val(self._keys_data_state_obligatory, self._values_object_data_state[1], 'Object2 data (state)')
        val(self._keys_data_covariance_obligatory, self._values_object_data_covariance[1], 'Object2 data (covariance)')

    def kvn(self, show_all=False):
        def append(s, d, d_obligatory):
            for k, v in d.items():
                k_str = k.ljust(37, ' ')
                if v is None:
                    if show_all or k in d_obligatory:
                        s += '{} =\n'.format(k_str)
                else:
                    if isinstance(v, float) or isinstance(v, int):
                        v_str = '{}'.format(v)
                        if 'e' in v_str:
                            v_str = '{:.3E}'.format(v)
                    else:
                        v_str = str(v)
                    s += '{} = {}\n'.format(k_str, v_str)
            return s
        ret = ''
        ret = append(ret, self._values_header, self._keys_header_obligatory)
        ret = append(ret, self._values_relative_metadata, self._keys_relative_metadata_obligatory)
        ret = append(ret, self._values_object_metadata[0], self._keys_metadata_obligatory)
        ret = append(ret, self._values_object_data_od[0], self._keys_data_od_obligatory)
        ret = append(ret, self._values_object_data_state[0], self._keys_data_state_obligatory)
        ret = append(ret, self._values_object_data_covariance[0], self._keys_data_covariance_obligatory)
        ret = append(ret, self._values_object_metadata[1], self._keys_metadata_obligatory)
        ret = append(ret, self._values_object_data_od[1], self._keys_data_od_obligatory)
        ret = append(ret, self._values_object_data_state[1], self._keys_data_state_obligatory)
        ret = append(ret, self._values_object_data_covariance[1], self._keys_data_covariance_obligatory)
        return ret

    def __repr__(self):
        return self.kvn()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __setitem__(self, key, value):
        if key in self._keys_header:
            self.set_header(key, value)
        elif key in self._keys_relative_metadata:
            self.set_relative_metadata(key, value)
        elif key.startswith('OBJECT1_'):
            key = key.split('OBJECT1_')[1]
            self.set_object(0, key, value)
        elif key.startswith('OBJECT2_'):
            key = key.split('OBJECT2_')[1]
            self.set_object(1, key, value)
        else:
            raise ValueError('Invalid key: {}'.format(key))


CDM = ConjunctionDataMessage
