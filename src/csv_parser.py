import pandas as pd
from datetime import datetime
from globals import RESOURCE_FOLDER

class CSVParser():

    def __init__(
        self, folderpath: str = RESOURCE_FOLDER
    ):
        self.folderpath = folderpath

    def parse_time(self,raw_time):
        try:
            # Define the format without the year
            date_format = "%m-%d %H:%M"
        
            # Parse the cleaned string into a datetime object
            parsed_datetime = datetime.strptime(raw_time, date_format)

            # Add placeholder year 2024.

            return parsed_datetime.replace(year = 2024)
        
        except ValueError:
            return None

    def retrieve_ais(self,path):
        df_ais = pd.read_csv(path, sep='|')
        df_ais['time'] = pd.to_datetime(df_ais['time'])
        df_ais['etaParsed'] = df_ais['etaRaw'].apply(self.parse_time)
        df_ais=df_ais.drop(['etaRaw'], axis = 1)
        return df_ais
    
    def retrieve_tests(self,path):
        df_tests = pd.read_csv(path, sep=',')
        df_tests["time"] = pd.to_datetime(df_tests["time"])
        return df_tests

    def retrieve_ports(self,path):
        df_ports = pd.read_csv(path, sep='|')
        df_ports['portLongitude'] = df_ports['longitude']
        df_ports['portLatitude'] = df_ports['latitude']
        df_ports=df_ports.drop(['name', 'portLocation', 'countryName', 'longitude', 'latitude'], axis = 1)
        return df_ports

    def retrieve_schedules(self,path):
        df_schedules = pd.read_csv(path, sep='|')
        df_schedules['sailingDate'] = pd.to_datetime(df_schedules['sailingDate'])
        df_schedules['arrivalDate'] = pd.to_datetime(df_schedules['arrivalDate'])

        df_schedules=df_schedules.drop(['shippingLineName', 'portName', 'portLongitude', 'portLatitude'], axis = 1)
        return df_schedules

    def retrieve_vessels(self,path):
        df_vessels = pd.read_csv(path, sep='|')
        df_vessels = df_vessels.drop(['vesselType', "depth", "draft", "maxWidth", "rampCapacity", "yearBuilt"], axis = 1)
        return df_vessels

    def retrieve_tests(self,path):
        df_tests = pd.read_csv(path, sep=',')
        df_tests["time"] = pd.to_datetime(df_tests["time"])
        return df_tests

    def retrieve_training_data(self):
        df_ais=self.retrieve_ais(self.folderpath+'/ais_train.csv')
        df_ports=self.retrieve_ports(self.folderpath+'/ports.csv')
        # df_schedules=self.retrieve_schedules(self.folderpath+'/schedules_to_may_2024.csv')
        # df_vessels=self.retrieve_vessels(self.folderpath+'/vessels.csv')
        

        result = df_ais
        result = pd.merge(df_ais, df_ports, on='portId')
        # result = pd.merge(result, df_vessels, on='vesselId')
        # result = pd.merge(result, df_schedules, on=['vesselId', 'portId'])
        #result = result.drop(['portId'], axis = 1)
        

        return result

    def retrieve_test_data(self):
        df_ais=self.retrieve_tests(self.folderpath+'/ais_test.csv')
        
        result = df_ais
        
        return result


if __name__ == '__main__':
    parser = CSVParser()
    parser.retrieve_training_data()
    