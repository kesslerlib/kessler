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
import numpy as np
import datetime

import kessler
import kessler.util


class UtilTestCase(unittest.TestCase):
    def test_tle(self):
        args = dict(
            satnum=43437,
            classification='U',
            international_designator='18100A',
            epoch_year=20,
            epoch_days=143.90384230,
            ephemeris_type=0,
            element_number=9996,
            revolution_number_at_epoch=56353,
            mean_motion=0.0011,
            mean_motion_first_derivative=6.9722e-13,
            mean_motion_second_derivative=.0,
            eccentricity=0.0221,
            inclination=1.7074,
            argument_of_perigee=2.1627,
            raan=4.3618,
            mean_anomaly=4.5224,
            b_star=0.0001)

        line1, line2 = kessler.util.tle(**args)
        line1Correct = '1 43437U 18100A   20143.90384230  .00041418  00000-0  10000-3 0 99968'
        line2Correct = '2 43437  97.8268 249.9127 0221000 123.9136 259.1144 15.12608579563539'

        self.assertEqual(line1Correct, line1)
        self.assertEqual(line2Correct, line2)

        args = dict(
            satnum=43437,
            classification='U',
            international_designator='18100A',
            epoch_year=20,
            epoch_days=143.90384230,
            ephemeris_type=0,
            element_number=9996,
            revolution_number_at_epoch=56353,
            mean_motion=0.0010,
            mean_motion_first_derivative=1.6261e-13,
            mean_motion_second_derivative=1e-5,
            eccentricity=0.0274,
            inclination=1.5086,
            argument_of_perigee=4.9757,
            raan=3.8719,
            mean_anomaly=6.0775,
            b_star=-0.0010)

        line1, line2 = kessler.util.tle(**args)
        line1Correct = '1 43437U 18100A   20143.90384230  .00009660  17108+9 -10000-2 0 99966'
        line2Correct = '2 43437  86.4364 221.8435 0274000 285.0866 348.2151 13.75098708563531'

        self.assertEqual(line1Correct, line1)
        self.assertEqual(line2Correct, line2)

    def test_from_datetime_to_cdm_datetime_str(self):
        date = datetime.datetime(2823, 3, 4, 12, 1, 23, 252 )
        date_str = kessler.util.from_datetime_to_cdm_datetime_str(date)
        date_str_correct = '2823-03-04T12:01:23.000252'
        self.assertEqual(date_str_correct, date_str)

    def test_from_jd_to_cdm_datetime_str(self):
        jd_date = 2906000.73344
        date_str = kessler.util.from_jd_to_cdm_datetime_str(jd_date)
        date_str_correct = '3244-04-04T05:36:09.216008'
        self.assertEqual(date_str_correct, date_str)

    def test_from_TEME_to_ITRF(self):
        state_TEME = np.array([[5094.18016210*1e3, 6127.64465950*1e3, 6380.34453270*1e3], [-4.746131487*1e3, 0.785818041*1e3, 5.531931288*1e3]])

        jd_date = 2433282.4235
        state_ITRF = kessler.util.from_TEME_to_ITRF(state_TEME, jd_date)
        state_ITRF_correct = np.array([np.array([ 7377976.66089733,-3010674.50719708, 6380344.5327]), np.array([-900.58420598,4224.28457883,5531.931288])])
        print('state_ITRF', state_ITRF)
        print('state_ITRF_correct', state_ITRF_correct)
        print('state_TEME', state_TEME)
        self.assertTrue(np.allclose(state_ITRF_correct, state_ITRF))
        self.assertAlmostEqual(np.linalg.norm(state_TEME[0]), np.linalg.norm(state_ITRF[0]), places=1)

    def test_from_cartesian_to_rtn(self):
        state_xyz = np.array([[1.0, 2.0, 3.4], [4.5, 6.2, 7.4]])
        state_rtn, cartesian_to_rtn_rotation_matrix = kessler.util.from_cartesian_to_rtn(state_xyz)
        rtn_to_cartesian_rotation_matrix = cartesian_to_rtn_rotation_matrix.T
        r_xyz, v_xyz = np.matmul(rtn_to_cartesian_rotation_matrix, state_rtn[0]), np.matmul(rtn_to_cartesian_rotation_matrix, state_rtn[1])

        self.assertTrue(np.allclose(state_xyz[0], r_xyz))
        self.assertTrue(np.allclose(state_xyz[1], v_xyz))
        self.assertAlmostEqual(np.linalg.norm(state_xyz[0]), np.linalg.norm(state_rtn[0]), places=1)
        self.assertAlmostEqual(np.linalg.norm(state_xyz[1]), np.linalg.norm(state_rtn[1]), places=1)

    def test_from_cartesian_to_rtn_2(self):
        state_xyz = np.array([[1.0, 2.0, 3.4], [4.5, 6.2, 7.4]])
        state_rtn, _ = kessler.util.from_cartesian_to_rtn(state_xyz)
        self.assertAlmostEqual(np.linalg.norm(state_xyz[0]), np.linalg.norm(state_rtn[0]), places=1)
        self.assertAlmostEqual(np.linalg.norm(state_xyz[1]), np.linalg.norm(state_rtn[1]), places=1)

    def test_getCcsdsTimeFormat(self):
        # This test is written by Andrew Ng, 19/03/22. It makes use of example CDMs provided by the NASA CARA
        # analysis repo at https://github.com/nasa/CARA_Analysis_Tools/tree/master/two-dimension_Pc/UnitTest/InputFiles.
        test_case1 = "2000-01-01T00:00:00.000" #From AlfanoTestCase11.cdm
        test_case2 = "2018-229T13:56:33.000" # From DensityDecorrelationTestCaseCDM.txt
        test_case1_correct = "yyyy-mm-ddTHH:MM:SS.FFF"
        test_case2_correct = "yyyy-DDDTHH:MM:SS.FFF"

        self.assertEqual(kessler.util.getCcsdsTimeFormat(test_case1), test_case1_correct)
        self.assertEqual(kessler.util.getCcsdsTimeFormat(test_case2), test_case2_correct) 
    def test_DOY2Date(self):
        # This test is written by Andrew Ng, 19/03/22. It makes use of example CDMs provided by the NASA CARA
        # analysis repo at https://github.com/nasa/CARA_Analysis_Tools/tree/master/two-dimension_Pc/UnitTest/InputFiles.
        example1 = "2010-202T12:25:19.000" # From SingleCovTestCase1-4.cdm
        example2 = "2018-229T13:56:33.000" # From DensityDecorrelationTestCaseCDM.txt
        DOY_1 = example1[5:5+3] 
        Year_1= example1[0:4]
        DOY_2 = example2[5:5+3]
        Year_2= example2[0:4]
        test_case1_correct = "2010-7-21T12:25:19.00"
        test_case2_correct = "2018-8-17T13:56:33.00"
        self.assertEqual(kessler.util.DOY2Date(example1, DOY_1, Year_1, 5), test_case1_correct)
        self.assertEqual(kessler.util.DOY2Date(example2, DOY_2, Year_2, 5), test_case2_correct) 

