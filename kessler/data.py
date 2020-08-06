import uuid
import os
import torch

from . import util
from .models import ConstellationPair
from .cdm import ConjunctionDataMessage


def generate_dataset(dataset_dir, num_events, *args, **kwargs):
    model = ConstellationPair(*args, **kwargs)
    def generate_conjunction():
        found = False
        while not found:
            trace = model.get_trace()
            if trace['conj']:
                found = True
        print('Generated event with {} CDMs'.format(len(trace['cdms'])))
        return trace

    print('Generating CDM dataset')
    print('Directory: {}'.format(dataset_dir))
    util.create_path(dataset_dir, directory=True)
    for i in range(num_events):
        print('Generating event {} / {}'.format(i+1, num_events))
        file_name_event = os.path.join(dataset_dir, 'event_{}'.format(str(uuid.uuid4())))
        
        trace = generate_conjunction()
        file_name_trace = file_name_event + '.trace'
        print('Saving trace: {}'.format(file_name_trace))
        torch.save(trace, file_name_trace)

        cdms = trace['cdms']
        for j, cdm in enumerate(cdms):
            file_name_suffix = '{}'.format(j).rjust(len('{}'.format(len(cdms))), '0')
            file_name_cdm = file_name_event + '_{}.cdm.kvn.txt'.format(file_name_suffix)
            print('Saving cdm  : {}'.format(file_name_cdm))
            cdm.save(file_name_cdm)
