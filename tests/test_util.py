import unittest
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
