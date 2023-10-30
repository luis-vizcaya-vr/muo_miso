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
        
        model_dict = {}
        if self.df_train.shape[0] < 700:
            print(f"{self.name} does not have enough data.")
        else:
            print(f"{self.name} has {self.df_train.shape[0]} training observations.")
            
        zonal_demand = self.df_train[["ts_hour_floor", "actuals_substation_total"]].merge(Simulator.zonal_demand_hist, how = "left", on = "ts_hour_floor")
        zonal_lmps = self.df_train[["ts_hour_floor", "actuals_substation_total"]].merge(Simulator.zonal_lmps_hist, how = "left", on = "ts_hour_floor")
        best_zonal_demand = self.find_max_corr(zonal_demand, "actuals_substation_total")
        zonal_demand = zonal_demand[["ts_hour_floor", best_zonal_demand]]
        best_zonal_lmp = self.find_max_corr(zonal_lmps, "actuals_substation_total")
        zonal_lmps = zonal_lmps[["ts_hour_floor", best_zonal_lmp]]
        X_train = self.df_train.merge(zonal_demand, how = "left", on = "ts_hour_floor")
        X_train = X_train.merge(zonal_lmps, how = "left", on = "ts_hour_floor")
        scaler = StandardScaler()
        X_train = X_train.drop(columns = drop_cols)
        X_train = X_train.fillna(X_train.median())
        model_dict["medians"] = X_train.median()
        cols = X_train.columns.to_list()
        model_dict["ordered_columns"] = cols
        X_train = scaler.fit_transform(X_train)
        model_dict["scaler"] = scaler
        y_train = self.df_train[actual_col]
        
        if self.zero_band is not None:
            y_train = np.where(y_train < self.zero_band, 0, y_train)
        
        model_dict["maximum"] = y_train.max()
        parameters = {"n_estimators" : [100], "min_samples_leaf" : [1], "min_samples_split" : [2], "max_depth" : [None]}
        clf = GridSearchCV(RandomForestRegressor(), parameters, scoring = "neg_mean_squared_error")
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        model_dict["model"] = best_model
        
        with open(f"C:/Users/i32213/OneDrive - Wood Mackenzie Limited/Desktop/MUO/muov2-training/debug/individual models/model_{self.name}_{model_run_date}.pkl", "wb") as f:
                    pickle.dump(model_dict, f)

        return model_dict
    

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
        merged_df = merged_df.merge(renew_meta, how = "outer", on = ["ts_hour_floor", "Substation"])
        return merged_df
    
    def get_train_test(self, split = True):
        
        # MERGE RENEWABLE FORECASTS AND METADATA
        renew_meta = self._merge_renew_metadata()
        
        dates_voltages = self._merge_dates_voltages(self._get_complete_date_range())
        
        # MERGE ACTUALS AND METADATA
        actuals_meta = self._merge_actuals_metadata()
        
        # MERGE VOLTAGES AND ACTUALS
        merged_df = self._merge_voltages_actuals(dates_voltages, actuals_meta)
        
        # MERGE LMPS
        merged_df = self._merge_lmps(merged_df)
        
        # MERGE RENEWABLE FORECASTS
        merged_df = self._merge_renew(merged_df, renew_meta)
        
        merged_df = merged_df[["ts_hour_floor", "Substation", "GeneratorName", "ResourceNode", "voltage", "LMP", "renew_forecast_substation_total", "actuals_substation_total"]]
        
        if split == True:
            df_train = merged_df[(merged_df.ts_hour_floor >= self.start_date) & (merged_df.ts_hour_floor < self.cutoff_date) & (merged_df.GeneratorName.notnull())]
            
            df_test = merged_df[(merged_df.ts_hour_floor >= self.cutoff_date) & (merged_df.ts_hour_floor <= self.end_date) & (merged_df.GeneratorName.notnull())]
            
            return df_train, df_test
        else:
            return merged_df
    
    
    def set_sub_dates(self, start_date, cutoff_date, end_date):
        
        self.start_date = start_date
        self.cutoff_date = cutoff_date
        self.end_date = end_date
             
    def set_sub_df_train_test(self, df):
        
        df_train, df_test = df
        self.df_train = df_train[["ts_hour_floor", "Substation", "voltage", "LMP", "renew_forecast_substation_total", "actuals_substation_total"]]
        self.df_test = df_test[["ts_hour_floor", "Substation", "voltage", "LMP", "renew_forecast_substation_total", "actuals_substation_total"]]
    
        
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
 
        