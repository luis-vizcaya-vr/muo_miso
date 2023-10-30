import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

class Substation:    
    def __init__(self, name = None):
        
        if name is None:
            self.name = None
        else:
            self.name = name.upper()
            
        self.actuals = None
        self.capacity = None
        self.cutoff_date = None
        self.df_train = None
        self.df_test = None
        self.end_date = None
        self.exclude_period = None
        self.fuel_type = None
        self.lmps = None
        self.metadata = None
        self.renew_forecast = None
        self.start_date = None
        self.unit_id = None
        self.voltages = None
        self.zero_band = None
        self.zonal_demand = None
        self.zonal_lmps = None
        self.zonal_solar = None
        

    # we can set the target_col equals actuals    
    def find_max_corr(df, target_col):
        df_copy = df.copy()
        df_copy.dropna(inplace = True)
        df_copy = df_copy[df_copy["actuals_substation_total"] != 0]
        
        if "ts_hour_floor" in df_copy.columns:
            df_copy.drop("ts_hour_floor", axis = 1, inplace = True)
        elif "index" in df_copy.columns:
            df_copy.drop("index", axis = 1, inplace = True)
            
        corr = df_copy.corr()[target_col]
        corr = corr.drop(target_col)
        corr = corr.abs()
        max_corr_col = corr.idxmax()    
        return max_corr_col
    
        
    def train_model(self, drop_cols = None, actual_col = None):
         pass

    def merge_renew_metadata(self):
        renew_meta = self.renew_forecast.merge(self.metadata, how = "left", on = "unit_id")
        renew_summed = renew_meta.groupby(["ts_hour_floor", "Substation"]).renew_forecast.sum()
        renew_summed = renew_summed.reset_index()
        renew_summed.rename(columns = {"renew_forecast" : "renew_forecast_substation_total"}, inplace = True)
        return renew_summed
    
    
    def get_complete_date_range(self):
        
        date_range = pd.date_range(self.start_date, self.end_date, freq = "1H")
        date_range_df = pd.DataFrame(date_range)
        date_range_df.rename(columns = {0 : "ts_hour_floor"}, inplace = True)
        
        return date_range_df
    
    
    def merge_date_range_with_indices(self):
        pass
    
    
    def merge_dates_voltages(self, date_range_df):
        dates_voltages = date_range_df.merge(self.voltages, how = "left", on = ["ts_hour_floor"])    
        return dates_voltages
    
    def merge_actuals_metadata(self):
        metadata = self.metadata[["ResourceNode", "Substation", "BidMapping"]].drop_duplicates()
        actuals_meta = self.actuals.merge(metadata, how = "left", on = "BidMapping")
        actuals_summed = actuals_meta.groupby(["ts_hour_floor", "Substation"]).actual.sum()
        actuals_summed = actuals_summed.reset_index()
        actuals_summed.rename(columns={"actual": "actuals_substation_total"}, inplace=True)
        return actuals_summed
    
    
    def merge_voltages_actuals(dates_voltages, actuals_meta):    
        merged_df = actuals_meta.merge(dates_voltages, how = "left", on = ["ts_hour_floor", "Substation"])
        return merged_df
    
    def merge_lmps(self, merged_df):
        merged_df = merged_df.merge(self.lmps, how = "outer", on = ["ts_hour_floor"])
        return merged_df
    
    def merge_renew(merged_df, renew_meta):
        pass

    def get_train_test(self, split = True):
        pass
    
    def set_sub_dates(self, start_date, cutoff_date, end_date):
        self.start_date = start_date
        self.cutoff_date = cutoff_date
        self.end_date = end_date
             
    def set_sub_df_train_test(self, df):
        pass
        
    def set_sub_exclude_period(self, exclude_period):    
        self.exclude_period = exclude_period
    
    def set_sub_zero_band(self, zero_band):
        self.zero_band = zero_band
    
    def set_sub_metadata(self, sub_metadata):    
        metadata, unit_id, capacity, unit_type = sub_metadata
        self.metadata = metadata
        self.unit_id = unit_id
        self.capacity = capacity
        self.fuel_type = unit_type
        
    def set_sub_lmps_hist(self, sub_lmps_hist):
        self.lmps = sub_lmps_hist
        
    def set_sub_voltage_hist(self, sub_voltage_hist):
        self.voltages = sub_voltage_hist
    
    def set_sub_actuals(self, sub_actuals):
        self.actuals = sub_actuals
    
    def set_sub_renew_hist(self, sub_renew_hist):
        self.renew_forecast = sub_renew_hist
 
        