import numpy as np
import warnings
import datetime
import uuid


# Based on CCSDS 508.0-B-1
# https://public.ccsds.org/Pubs/508x0b1e2c1.pdf
class CDM():
    def __init__(self):
        self._obligatory = {}
        self._optional = {}

        # Header
        # Relative metadata
        # Metadata
        # Data (for Object1 and Object2)
        #  OD, State, Covariance
        # Optional comments

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

        self.set_header('CCSDS_CDM_VERS', '1.0')
        self.set_header('ORIGINATOR', 'ESA_FDL_CONSTELLATIONS')
        self.set_header('CREATION_DATE', datetime.datetime.utcnow().isoformat())
        self.set_header('MESSAGE_ID', 'ESA_FDL_{}'.format(str(uuid.uuid1())))

        self.set_object(0, 'OBJECT', 'OBJECT1')
        self.set_object(0, 'INTERNATIONAL_DESIGNATOR', 'UNKNOWN')
        self.set_object(0, 'OBJECT_NAME', 'ESA_FDL_TARGET')
        self.set_object(0, 'CATALOG_NAME', 'UNKNOWN')
        self.set_object(0, 'EPHEMERIS_NAME', 'NONE')
        self.set_object(0, 'COVARIANCE_METHOD', 'CALCULATED')
        self.set_object(0, 'MANEUVERABLE', 'N/A')
        self.set_object(0, 'REF_FRAME', 'ITRF')

        self.set_object(1, 'OBJECT', 'OBJECT2')
        self.set_object(1, 'INTERNATIONAL_DESIGNATOR', 'UNKNOWN')
        self.set_object(1, 'OBJECT_NAME', 'ESA_FDL_CHASER')
        self.set_object(1, 'CATALOG_NAME', 'UNKNOWN')
        self.set_object(1, 'EPHEMERIS_NAME', 'NONE')
        self.set_object(1, 'COVARIANCE_METHOD', 'CALCULATED')
        self.set_object(1, 'MANEUVERABLE', 'N/A')
        self.set_object(1, 'REF_FRAME', 'ITRF')

    def set_header(self, key, value):
        if key in self._keys_header:
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

        miss_distance = np.linalg.norm(state_object1[0]-state_object2[0])
        self.set_relative_metadata('MISS_DISTANCE', miss_distance)

    def _update_state_relative(self):
        def uvw_matrix(r, v):
            u = r / np.linalg.norm(r)
            w = np.cross(r, v)
            w = w / np.linalg.norm(w)
            v = np.cross(w, u)
            return np.vstack((u, v, w))

#         def from_cartesian_to_rtn(r, v):
#             T = uvw_matrix(r, v)
#             r_rtn = np.dot(T,r)
#             v_rtn = np.dot(T,v)
#             return r_rtn, v_rtn
#         def relative_state(state_object1, state_object2):
#             return np.zeros([2, 3])
        def relative_state(state_obj_1, state_obj_2):
            rot_matrix=uvw_matrix(state_obj_1[0], state_obj_1[1])
            rel_position_xyz = state_obj_2[0] - state_obj_1[0]
            rel_velocity_xyz = state_obj_2[1] - state_obj_1[1]
            relative_state=np.zeros([2,3])
            relative_state[0] = np.array([np.dot(rot_matrix[0],rel_position_xyz), np.dot(rot_matrix[1],rel_position_xyz), np.dot(rot_matrix[2],rel_position_xyz)])
            relative_state[1] = np.array([np.dot(rot_matrix[0],rel_velocity_xyz), np.dot(rot_matrix[1],rel_velocity_xyz), np.dot(rot_matrix[2],rel_velocity_xyz)])
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
                    s += '{} = {}\n'.format(k_str, v)
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
