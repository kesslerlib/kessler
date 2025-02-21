import numpy as np


class ObservationModel():
    def __init__(self, instrument_characteristics = {}):
        self._instrument_characteristics=instrument_characteristics

    def __repr__(self):
        return 'ObservationModel'

    def observe(self, time, spacecraft):
        raise NotImplementedError()


class GNSS(ObservationModel):
    def __init__(self, instrument_characteristics = {'bias_xyz': np.zeros((2,3)), 'covariance_rtn': np.array([1e-9, 0.00115849341564346, 0.000059309835843067, 1e-9, 1e-9, 1e-9])**2}):
        super().__init__(instrument_characteristics)

    def __repr__(self):
        return('GNNS({})'.format(self._instrument_characteristics))

    def observe(self, time, spacecraft):
        state_xyz = spacecraft['state_xyz'] + self._instrument_characteristics['bias_xyz']
        covariance_rtn = self._instrument_characteristics['covariance_rtn']
        return state_xyz, covariance_rtn


class Radar(ObservationModel):
    def __init__(self, instrument_characteristics = {'bias_xyz': np.zeros((2,3)), 'covariance_rtn': np.array([20, 500, 1, 0.0001, 0.0001, 0.0001])**2}):
        super().__init__(instrument_characteristics)

    def __repr__(self):
        return('Radar({})'.format(self._instrument_characteristics))

    def observe(self, time, spacecraft):
        state_xyz = spacecraft['state_xyz'] + self._instrument_characteristics['bias_xyz']
        covariance_rtn = self._instrument_characteristics['covariance_rtn']
        return state_xyz, covariance_rtn
