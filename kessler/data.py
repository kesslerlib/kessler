import uuid
import os
import torch

from . import util
from .models import ConstellationPair
from .cdm import ConjunctionDataMessage


def generate_dataset(dataset_dir, num_events, *args, **kwargs):
    model = ConstellationPair(*args, **kwargs)

    print('Generating CDM dataset')
    print('Directory: {}'.format(dataset_dir))
    util.create_path(dataset_dir, directory=True)
    for i in range(num_events):
        print('Generating event {} / {}'.format(i+1, num_events))
        file_name_event = os.path.join(dataset_dir, 'event_{}'.format(str(uuid.uuid4())))
        
        trace = model.get_conjunction()
        file_name_trace = file_name_event + '.trace'
        print('Saving trace: {}'.format(file_name_trace))
        torch.save(trace, file_name_trace)

        cdms = trace['cdms']
        for j, cdm in enumerate(cdms):
            file_name_suffix = '{}'.format(j).rjust(len('{}'.format(len(cdms))), '0')
            file_name_cdm = file_name_event + '_{}.cdm.kvn.txt'.format(file_name_suffix)
            print('Saving cdm  : {}'.format(file_name_cdm))
            cdm.save(file_name_cdm)


# # ----- this is to read CDMs as pandas dataframe -----
# import pandas as pd
# #CDMs=pd.DataFrame()
# k=0
# #pd.concat([CDMs] * 23000)
# list_cdms=[]
# for i in sorted(os.listdir('/cdm_data/01_mt_5e3_mc_1e3_c_cov_min/')):
# k=k+1
# print(k,end='\r')
# cdm=kessler.cdm.ConjunctionDataMessage.load('/cdm_data/01_mt_5e3_mc_1e3_c_cov_min/'+i)
# list_cdms.append(cdm.as_dataframe())
# CDMs_1=pd.concat(list_cdms,ignore_index=True)
# #CDMkessler.cdm.ConjunctionDataMessage.load() 