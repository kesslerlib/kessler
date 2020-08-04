import uuid
import os

from . import util
from .models import ConstellationPair
from .cdm import ConjunctionDataMessage


def generate_dataset(dataset_dir, num_events, *args, **kwargs):
    model = ConstellationPair(*args, **kwargs)
    def generate_event():
        found = False
        while not found:
            trace = model.get_trace()
            if trace['conj']:
                found = True
        cdms = trace['cdms']
        print('Generated event with {} CDMs'.format(len(cdms)))
        return trace['cdms']

    print('Generating CDM dataset')
    print('Directory: {}'.format(dataset_dir))
    util.create_path(dataset_dir, directory=True)
    for i in range(num_events):
        print('Generating event {} / {}'.format(i+1, num_events))
        file_name_event = os.path.join(dataset_dir, 'event_{}'.format(str(uuid.uuid4())))
        cdms = generate_event()
        for j, cdm in enumerate(cdms):
            file_name_suffix = '{}'.format(j).rjust(len('{}'.format(len(cdms))), '0')
            file_name_cdm = file_name_event + '_{}'.format(file_name_suffix)
            print('Saving cdm: {}'.format(file_name_cdm))
            cdm.save(file_name_cdm)
