import pandas as pd

class Event():
    def __init__(self, cdms=[]):
        self._cdms = cdms

    def add(self, cdm):
        self._cdms.append(cdm)
    
    def as_dataframe(self):
        list_cdms=[]
        for cdm in self._cdms:
            list_cdms.append(cdm.as_dataframe())
        return pd.concat(list_cdms,ignore_index=True)

    def plot_uncertainty_evolution(self, name_of_feature, ax=None, square_root_kessler = True, label=None):
        data_x = []
        data_y=[]
        for cdm in range(len(self._cdms)):
            data_x.append(convert_date_to_days(cdm['TCA'],cdm['CREATION_DATE']))
            data_y.append(cdm[name_of_feature])
        # Creating axes instance 
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(data_x, data_y, alpha=0.3, label=label)
    
    def event_characteristics_plots(self, number_of_cdms):
        CDMs=self._cdms
        obj1_covariance_names = ["OBJECT1_CR_R", "OBJECT1_CR_T", "OBJECT1_CR_N", "OBJECT1_CR_RDOT",  "OBJECT1_CR_TDOT", "OBJECT1_CR_NDOT",
                                "OBJECT1_CT_T", "OBJECT1_CT_N", "OBJECT1_CT_RDOT", "OBJECT1_CT_TDOT" , "OBJECT1_CT_NDOT",
                                "OBJECT1_CN_N", "OBJECT1_CN_RDOT", "OBJECT1_CN_TDOT", "OBJECT1_CN_NDOT",
                                "OBJECT1_CRDOT_RDOT","OBJECT1_CRDOT_TDOT","OBJECT1_CRDOT_NDOT",
                                "OBJECT1_CTDOT_TDOT","OBJECT1_CTDOT_NDOT",
                                "OBJECT1_NDOT_NDOT"]
        
        obj2_covariance_names = ["OBJECT2_CR_R", "OBJECT2_CR_T", "OBJECT2_CR_N", "OBJECT2_CR_RDOT",  "OBJECT2_CR_TDOT", "OBJECT2_CR_NDOT",
                                "OBJECT2_CT_T", "OBJECT2_CT_N", "OBJECT2_CT_RDOT", "OBJECT2_CT_TDOT" , "OBJECT2_CT_NDOT",
                                "OBJECT2_CN_N", "OBJECT2_CN_RDOT", "OBJECT2_CN_TDOT", "OBJECT2_CN_NDOT",
                                "OBJECT2_CRDOT_RDOT","OBJECT2_CRDOT_TDOT","OBJECT2_CRDOT_NDOT",
                                "OBJECT2_CTDOT_TDOT","OBJECT2_CTDOT_NDOT",
                                "OBJECT2_NDOT_NDOT"]

        fig, axs = plt.subplots(6, 6, figsize=(20,5))

        for i, string in enumerate(obj1_covariance_names):
            ax=axs[i]
            plot_uncertainty_evolution(string, "OBJECT1_OBJECT_DESIGNATOR",False, ax=ax)
            ax.set_title(item)
        plt.tight_layout()

    def __repr__(self):
        return 'Event(cdms:{})'.format(len(self._cdms))

    def __getitem__(self, i):
        return self._cdms[i]

    def __len__(self):
        return len(self._cdms)