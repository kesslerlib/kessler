# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
# Trillium Technologies
# Giacomo Acciarini
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.


import unittest
import tempfile
import uuid
import os
import numpy as np

from kessler import ConjunctionDataMessage


class CDMTestCase(unittest.TestCase):
    def test_cdm_load_save_load(self):
        file_content = """
            CCSDS_CDM_VERS                        = 1.0
            CREATION_DATE                         = 2013-01-09T20:59:56.000
            ORIGINATOR                            = JSpOC
            MESSAGE_FOR                           = IRIDIUM 26
            MESSAGE_ID                            = 2013009205956Z
            TCA                                   = 2013-01-10T13:22:45.117
            MISS_DISTANCE                         = 180.0
            RELATIVE_SPEED                        = 8111.0
            RELATIVE_POSITION_R                   = 35.9
            RELATIVE_POSITION_T                   = -148.8
            RELATIVE_POSITION_N                   = 95.7
            RELATIVE_VELOCITY_R                   = -4.1
            RELATIVE_VELOCITY_T                   = -4421.4
            RELATIVE_VELOCITY_N                   = -6800.2
            OBJECT                                = OBJECT1
            OBJECT_DESIGNATOR                     = 24903.0
            CATALOG_NAME                          = US SATCAT
            OBJECT_NAME                           = IRIDIUM 26
            INTERNATIONAL_DESIGNATOR              = 1997-043A
            EPHEMERIS_NAME                        = NONE
            COVARIANCE_METHOD                     = CALCULATED
            MANEUVERABLE                          = N/A
            ORBIT_CENTER                          = EARTH
            REF_FRAME                             = ITRF
            GRAVITY_MODEL                         = EGM-96: 36D 36O
            ATMOSPHERIC_MODEL                     = JACCHIA 70 DCA
            SOLAR_RAD_PRESSURE                    = NO
            EARTH_TIDES                           = NO
            INTRACK_THRUST                        = NO
            RECOMMENDED_OD_SPAN                   = 6.58
            ACTUAL_OD_SPAN                        = 6.58
            OBS_AVAILABLE                         = 573.0
            OBS_USED                              = 571.0
            RESIDUALS_ACCEPTED                    = 98.1
            WEIGHTED_RMS                          = 0.944
            CD_AREA_OVER_MASS                     = 0.035005
            CR_AREA_OVER_MASS                     = 0.0
            SEDR                                  = 7.305E-05
            X                                     = -731.972155846
            Y                                     = -1871.205777005
            Z                                     = -6876.444217211
            X_DOT                                 = -1.117777619
            Y_DOT                                 = -7.044089359
            Z_DOT                                 = 2.036664419
            CR_R                                  = 38.33
            CT_R                                  = 93.6
            CT_T                                  = 3410.0
            CN_R                                  = -13.06
            CN_T                                  = 2.131
            CN_N                                  = 93.39
            CRDOT_R                               = 0.0
            CRDOT_T                               = 0.0
            CRDOT_N                               = 0.0
            CRDOT_RDOT                            = 0.0
            CTDOT_R                               = 0.0
            CTDOT_T                               = 0.0
            CTDOT_N                               = 0.0
            CTDOT_RDOT                            = 0.0
            CTDOT_TDOT                            = 0.0
            CNDOT_R                               = 0.0
            CNDOT_T                               = 0.0
            CNDOT_N                               = 0.0
            CNDOT_RDOT                            = 0.0
            CNDOT_TDOT                            = 0.0
            CNDOT_NDOT                            = 0.0
            OBJECT                                = OBJECT2
            OBJECT_DESIGNATOR                     = 33759.0
            CATALOG_NAME                          = US SATCAT
            OBJECT_NAME                           = COSMOS 2251 DEB
            INTERNATIONAL_DESIGNATOR              = 1993-036G
            EPHEMERIS_NAME                        = NONE
            COVARIANCE_METHOD                     = CALCULATED
            MANEUVERABLE                          = N/A
            ORBIT_CENTER                          = EARTH
            REF_FRAME                             = ITRF
            GRAVITY_MODEL                         = EGM-96: 36D 36O
            ATMOSPHERIC_MODEL                     = JACCHIA 70 DCA
            SOLAR_RAD_PRESSURE                    = YES
            EARTH_TIDES                           = NO
            INTRACK_THRUST                        = NO
            RECOMMENDED_OD_SPAN                   = 6.06
            ACTUAL_OD_SPAN                        = 6.06
            OBS_AVAILABLE                         = 180.0
            OBS_USED                              = 180.0
            RESIDUALS_ACCEPTED                    = 99.5
            WEIGHTED_RMS                          = 1.426
            CD_AREA_OVER_MASS                     = 0.20058
            CR_AREA_OVER_MASS                     = 0.087953
            SEDR                                  = 0.000525006
            X                                     = -732.050575868
            Y                                     = -1871.058618906
            Z                                     = -6876.513305974
            X_DOT                                 = 6.170235153
            Y_DOT                                 = -3.879972556
            Z_DOT                                 = 0.404182951
            CR_R                                  = 408.7
            CT_R                                  = -535.0
            CT_T                                  = 77970.0
            CN_R                                  = -17.21
            CN_T                                  = -79.98
            CN_N                                  = 239.9
            CRDOT_R                               = 0.0
            CRDOT_T                               = 0.0
            CRDOT_N                               = 0.0
            CRDOT_RDOT                            = 0.0
            CTDOT_R                               = 0.0
            CTDOT_T                               = 0.0
            CTDOT_N                               = 0.0
            CTDOT_RDOT                            = 0.0
            CTDOT_TDOT                            = 0.0
            CNDOT_R                               = 0.0
            CNDOT_T                               = 0.0
            CNDOT_N                               = 0.0
            CNDOT_RDOT                            = 0.0
            CNDOT_TDOT                            = 0.0
            CNDOT_NDOT                            = 0.0
            """
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        with open(file_name, 'w') as f:
            f.write(file_content)

        cdm = ConjunctionDataMessage.load(file_name)

        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        cdm.save(file_name)

        cdm = ConjunctionDataMessage.load(file_name)

        cdm_covariance_1 = cdm.get_covariance(0)
        cdm_covariance_2 = cdm.get_covariance(1)
        cdm_covariance_1_correct = np.array([[3.833e+01, 9.360e+01, -1.306e+01, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [9.360e+01, 3.410e+03, 2.131e+00, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [-1.306e+01, 2.131e+00, 9.339e+01, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00]])
        cdm_covariance_2_correct = np.array([[4.087e+02, -5.350e+02, -1.721e+01, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [-5.350e+02, 7.797e+04, -7.998e+01, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [-1.721e+01, -7.998e+01, 2.399e+02, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
                                             [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00]])

        self.assertEqual(cdm_covariance_1_correct.tolist(), cdm_covariance_1.tolist())
        self.assertEqual(cdm_covariance_2_correct.tolist(), cdm_covariance_2.tolist())
