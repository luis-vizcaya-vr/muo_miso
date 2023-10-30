import sys
import pathlib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

sys.path.append('../')

class Simulator:
    
    data_folder = r"..\data"

    def __init__(self, sub_list = None, metadata = None):
        
        if sub_list is None:
            self.sub_list = []
        else:
            self.sub_list = sub_list
        
        self.metadata = None
        self.actuals = None
        self.lmps_hist = None
        self.zonal_lmps_hist = None
        self.voltage_hist = None
        self.renew_hist = None
        self.monitored_hist = None
        
    
    def create_subs(subs):
        #create a list of subs object in the class simulator with the names given in subs
        pass

    def get_actuals(self, file_list = []):
        #get actuals for all substation from a list of files
        pass

    def update_actuals_subs(self, subs):
        pass

    def update_subs_actuals(self, subs = None):
        #update specific list of subs with the actuals in self.actuals,
        #if subs is not specified, it updates all the subs
        pass
    
    def get_zonal_demand_hist(self, file_list = []):
        #get historical demand from a file
        pass

    def update_zonal_demand_hist_subs(self, subs = None):
        #udpate zonal demand for subs, in case subs in not specified, it'll update all subs
        pass

    def get_zonal_lmps_hist(self, file_list = []):        
        #get historical zonal lmps from a file, in case subs in not specified, it'll update all subs
        pass    

    def udpate_zonal_lmps_hist_subs(self, subs):
        #update subs zonal lmps, in case subs in not specified, it'll update all subs
        pass
        
    def get_metadata(self, file_list = []):
        #update metadata from a file
        pass
        
    def update_metadata_subs(self, subs = None):
        #update subs metadata, in case subs is not specified, it'll update all subs
        pass

    def get_lmps_hist(self, file_list = []):
        #get historical lmps from a list of files
        pass
        
    def update_lmps_hist_subs(self, subs = None):
        #update subs historical lmps, in case subs in not specified, it'll update all subs
        pass

    def get_voltage_hist(self, file_list = []):
        # get historical lmps from a list of files
        pass

    def update_voltage_hist_subs(self, subs = None):
        # update subs historical lmps, in case subs in not specified, it'll update all subs
        pass
    
    def get_renew_hist(self, file_list = []):
        #update renew forecast from a file list
        pass

    def get_renew_hist_subs(self, subs):    
        #update subs historical renew,def update_actuals_subs(self, subs):
        pass
