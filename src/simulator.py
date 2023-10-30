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
        
    @staticmethod
    def __add_ts_hour_floor_col(df, ts_col_name, method = "round"):
        if method == "round":
            df.loc[:, "ts_hour_floor"] = df[str(ts_col_name)].dt.round("H")
        elif method == "floor":
            df.loc[:, "ts_hour_floor"] = df[str(ts_col_name)].dt.round("60min")
        else:
            raise ValueError("Method neither round nor floor")
        return df
            
    def _get_metadata(self, metadata_folder = "/metadata/", filename_list = ["metadata_20230913.csv", "MappingMUO-PowerRT_20230915.csv"]):
        
        metadata = pd.read_csv(Simulator.data_folder + metadata_folder + filename_list[0])
        revised_metadata = pd.read_csv(Simulator.data_folder + metadata_folder + filename_list[1])
        
        revised_metadata["monitored"] = np.where(revised_metadata["Monitored"].str[:3] == "Yes", True, False)
        monitored_count = revised_metadata.groupby("Substation").monitored.nunique()
        
        substations_where_units_are_all_same_type = monitored_count[monitored_count == 1].index
        revised_metadata["monitored"] = np.where(revised_metadata.Substation.isin(substations_where_units_are_all_same_type) 
                                                            & (~revised_metadata["monitored"]),
                                                            False, True)
        unmonitored_subs = revised_metadata[~revised_metadata.monitored].Substation.unique()
        metadata["monitored"] = np.where(metadata.Substation.isin(unmonitored_subs), False, True)
        metadata.rename(columns = {"SeerUnitId" : "unit_id"}, inplace = True)
        
        self.metadata = metadata
        
    def _get_sub_metadata(self, Substation):
        
        metadata = self.metadata
        metadata = metadata[metadata["Substation"].str.contains(Substation.name)]
        
        return metadata, metadata["unit_id"].unique()[0], metadata["Capacity"].unique()[0], metadata["UnitType"].unique()[0]
    
    def _get_lmps_hist(self, lmp_folder = "/raw/lmp/", filename_list = ["lmp_hourly_20220126_20220331_20220930_QA.parquet", "lmp_hourly_20220401_20220531_20220930_QA.parquet", "lmp_hourly_20220601_20220731_20220930_QA.parquet", "lmp_hourly_20220801_20220922_20220930_QA.parquet", "lmp_hourly_20220923_20221231_20230426_QA.parquet", "lmp_hourly_20230101_20230615_20230616_QA.parquet"]):
        
        lmp_file_list = []
        
        for i in range(len(filename_list)):
            lmp_file_list.append(pd.read_parquet(Simulator.data_folder + lmp_folder + filename_list[i]))
            
        lmps = pd.concat(lmp_file_list)
        
        lmps = lmps.sort_values(by = ["SettlementPoint", "SCEDTimestamp"])
        lmps["study_start_date"] = pd.to_datetime(lmps.SCEDTimestamp.dt.date)
        lmps = Simulator.__add_ts_hour_floor_col(lmps, "SCEDTimestamp")
        
        lmps = lmps.groupby(["ts_hour_floor", "SettlementPoint", "study_start_date"]).LMP.first().reset_index()
        lmps.rename(columns = {"SettlementPoint" : "ResourceNode"}, inplace = True)
        
        self.lmps_hist = lmps
        
    def _get_sub_lmps_hist(self, Substation):
        
        lmps = self.lmps_hist
        lmps = lmps[lmps["ResourceNode"].str.contains(Substation.name)]
        lmps = lmps[["ts_hour_floor", "ResourceNode", "LMP"]]
        
        return lmps
    
    def _get_voltage_hist(self, voltage_folder = "/raw/voltages_mapped/", filename_list = ["voltage_mapped_v2_20220126_20220331_20220930_QA.parquet", "voltage_mapped_v2_20220401_20220531_20220930_QA.parquet", "voltage_mapped_v2_20220601_20220731_20220930_QA.parquet", "voltage_mapped_v2_20220801_20220922_20220930_QA.parquet", "voltage_mapped_v2_20220923_20221231_20230426_QA.parquet", "voltage_mapped_v2_20230101_20230424_20230426_QA.parquet"]):
        
        volt_file_list = []
        
        for i in range(len(filename_list)):
            volt_file_list.append(pd.read_parquet(Simulator.data_folder + voltage_folder + filename_list[i]))
            
        voltages = pd.concat(volt_file_list)
        
        voltages = voltages.drop_duplicates()
        voltages.rename(columns = {"STATION_NAME" : "Substation"}, inplace = True)

        voltages["SE_EXECUTION_TIME"] = pd.to_datetime(voltages["SE_EXECUTION_TIME"])
        voltages.sort_values(by = ["SE_EXECUTION_TIME", "Substation", "GeneratorName"], inplace = True)
        voltages = Simulator.__add_ts_hour_floor_col(voltages, "SE_EXECUTION_TIME")
        voltages.rename(columns = {"VOLTAGE_ESTIMATE" : "voltage"}, inplace = True)
        
        self.voltage_hist = voltages
        
            
    def _get_sub_voltage_hist(self, Substation):
     
        voltages = self.voltage_hist
        voltages = voltages[voltages["Substation"].str.contains(Substation.name)]
        voltages = voltages[["ts_hour_floor", "Substation", "GeneratorName", "voltage"]]
        
        return voltages
    
    def _get_actuals(self, actuals_folder = "/raw/actuals/", filename_list = ["actuals_20220126_20220331_20221011_QA.parquet", "actuals_20220401_20220531_20221011_QA.parquet", "actuals_20220601_20220805_20221011_QA.parquet", "actuals_20220806_20220902_20221110_QA.parquet", "actuals_20220903_20221031_20230206_QA.parquet", "actuals_20221101_20230224_20230606_QA.parquet", "actuals_20230225_20230408_20230616_QA.parquet", "actuals_20230425_20230609_20230925_QA.parquet"]):
        
        actual_file_list = []
        
        for i in range(len(filename_list)):
            actual_file_list.append(pd.read_parquet(Simulator.data_folder + actuals_folder + filename_list[i]))
            
        actuals = pd.concat(actual_file_list)
        
        actuals.drop_duplicates(subset = ["SCEDTimestamp", "ResourceName"], inplace = True)
        
        actuals["repeated_hour_flag"] = "N"
        actuals["SCEDTimestamp"] = pd.to_datetime(actuals["SCEDTimestamp"])
        actuals = Simulator.__add_ts_hour_floor_col(actuals, "SCEDTimestamp", method = "floor")
        actuals = actuals[actuals["repeated_hour_flag"] == "N"]
        actuals = actuals.groupby(["ts_hour_floor", "ResourceName"]).ActualOutput.first().reset_index()
        actuals.rename(columns = {"ResourceName" : "BidMapping", "ActualOutput" : "actual"}, inplace = True)
        
        self.actuals = actuals
    
    def _get_sub_actuals(self, Substation):
        
        actuals = self.actuals
        actuals = actuals[actuals["BidMapping"].str.contains(Substation.name)]
        actuals = actuals[["ts_hour_floor", "BidMapping", "actual"]]
        
        return actuals
    
    
    def _get_renew_hist(self, renew_folder = "/raw/renewable_forecasts/", filename_list = ["renew_forecasts_20220126_20220331_20220930_archive_Prod.parquet", "renew_forecasts_20220401_20220531_20220930_archive_Prod.parquet", "renew_forecasts_20220601_20220825_20220930_archive_Prod.parquet", "renew_forecasts_20220826_20220922_20220930_archive_Prod.parquet", "renew_forecasts_20220923_20221231_20230606_archive_Prod.parquet", "renew_forecasts_20230101_20230224_20230606_archive_Prod.parquet", "renew_forecasts_20230225_20230424_20230616_archive_Prod.parquet", "renew_forecasts_20230425_20230609_20230925_archive_Prod.parquet"]):
        
        renew_file_list = []
        
        for i in range(len(filename_list)):
            renew_file_list.append(pd.read_parquet(Simulator.data_folder + renew_folder + filename_list[i]))
            
        renew = pd.concat(renew_file_list)
        
        renew["Date"] = pd.to_datetime(renew["Date"])
        renew = renew[renew.Date.dt.minute == 0].copy()
        
        renew.rename(columns = {"category_id" : "unit_id", "Date" : "ts_hour_floor", "Value" : "renew_forecast"}, inplace = True)
        
        self.renew_hist = renew
        
    def _get_sub_renew_hist(self, Substation):
        
        renew = self.renew_hist
        renew = renew[renew["unit_id"] == Substation.unit_id]
        renew = renew[["unit_id", "ts_hour_floor", "renew_forecast"]]
        
        return renew
    
    def _get_zonal_demand_hist(self, demand_folder = "/raw/demand/", filename_list = ["ercot_demand_20220101_20230925_20230926.parquet"]):
        
        demand = pd.read_parquet(Simulator.data_folder + demand_folder + filename_list[0])
        demand.reset_index(inplace = True)
        demand.rename(columns = {"DateEntered" : "ts_hour_floor"}, inplace = True)
        demand["ts_hour_floor"] = pd.to_datetime(demand["ts_hour_floor"])
        
        self.zonal_demand_hist = demand
    
    def _get_zonal_lmps_hist(self, zonal_demand_folder = "/raw/zonal_lmps/", filename_list = ["ercot_zonal_lmps_20220101_20230925_20230926.parquet"]):
        
        zonal_lmps = pd.read_parquet(Simulator.data_folder + zonal_demand_folder + filename_list[0])
        zonal_lmps.reset_index(inplace = True)
        zonal_lmps.rename(columns = {"DateEntered" : "ts_hour_floor"}, inplace = True)
        zonal_lmps["ts_hour_floor"] = pd.to_datetime(zonal_lmps["ts_hour_floor"])
        
        self.zonal_lmps_hist = zonal_lmps
        
    def _get_zonal_solar_hist(self, zonal_solar_folder = "/raw/solar/", filename_list = ["zonal_solar_ERCOT_20220906_20230811_20230830_QA.parquet"]):
        
        zonal_solar = pd.read_parquet(Simulator.data_folder + zonal_solar_folder + filename_list[0])
        zonal_solar["ts_hour_floor"] = pd.to_datetime(zonal_solar["ts_hour_floor"])
        
        self.zonal_solar_hist = zonal_solar