"""
Code for generating data for single-output and multiple-output Bayesian calibration
"""
__author__ = "Yaonan Gu, Chao Song"
__copyright__ = "Copyright 2023, National University of Singapore - NUS"
__credits__ = [""]
__license__ = "To be added"
__version__ = "3"
__maintainer__ = "Yaonan Gu, Chao Song"
__email__ = "yaonangu@u.nus.edu, nussongchao@gmail.com"
__status__ = "Experimental/Beta"


import os
import sys

import yaml
import numpy as np
import pandas as pd
from eppy import modeleditor
from eppy.modeleditor import IDF
import subprocess
import csv
import icecream as ic
import pickle
from datetime import datetime as dt


now = dt.now()
# Put the timestamp to identify the case easier
t = now.strftime("timestamp")  
np.random.seed(2)


class multi_data_simulator:
    def __init__(self, case_name, conf, report_freq, simulation_num):
        self.case_name = case_name
        self.conf = conf
        self.yc_keys = self.conf['yc_keys']
        self.vc_keys = self.conf['vc_keys']
        self.zone_keys = self.conf['zone_keys']
        self.xc_keys = self.conf['xc_keys']
        self.tc_keys = self.conf['tc_keys']
        self.report_freq = report_freq
        self.sm_num = simulation_num

    def LHSample(self, D,bounds,N):
        ''' 
        :type D: int, the number of parameters
        :type bounds: List[float], the range set for parameters
        :type N: int, the number of LHS layers
        :rtype: array, sampled data
        '''

        result = np.empty([N, D])
        temp = np.empty([N])
        d = 1.0 / N

        for i in range(D):

            for j in range(N):
                temp[j] = np.random.uniform(
                    low=j * d, high=(j + 1) * d, size = 1)[0]

            np.random.shuffle(temp)

            for j in range(N):
                result[j, i] = temp[j]

        # Scale the data
        b = np.array(bounds)
        lower_bounds = b[:,0]
        upper_bounds = b[:,1]
        if np.any(lower_bounds > upper_bounds):
            print('Wrong range!')
            return None

        np.add(np.multiply(result,
                           (upper_bounds - lower_bounds),
                           out=result),
               lower_bounds,
               out=result)
        self.LHS_result = result
        return self.LHS_result

    def comp_data_reader(self, eso_file):
        with open(eso_file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Eplusout = []
            for row in readCSV:
                Eplusout.append(row)

        yc_mtr_number=[]
        for i in range(len(self.yc_keys)):
            for row in Eplusout:
                if len(row)>3:
                    if self.yc_keys[i] in row[2]:
                        yc_mtr_number.append(row[0])


        yc_mtr_values=[]
        for i in range(len(yc_mtr_number)):
            yc_mtr_value=[]
            for row in Eplusout:
                if len(row)>3:                
                    if yc_mtr_number[i] == row[0]:
                        if self.yc_keys[i] not in row[2]:
                            yc_mtr_value.append(float(row[1])/3600000)
            yc_mtr_values.append(yc_mtr_value)


        # For indoor air temperature
        yc_var_number=[]
        for i in range(len(self.vc_keys)):
            for row in Eplusout:
                if len(row)>6:
                    if self.vc_keys[i] in row[3]:
                        yc_var_number.append(row[0])

        yc_var_values = []
        for i in range(len(yc_var_number)):
            yc_var_value=[]
            for row in Eplusout:
                if len(row)>6:
                    if yc_var_number[i] == row[0]:
                        if self.vc_keys[0] not in row[3]:
                            yc_var_value.append(float(row[1]))

            yc_var_values.append(yc_var_value)

        xc_mtr_number=[]
        for i in range(len(self.xc_keys)):
            for row in Eplusout:
                if len(row)>3:
                    if self.xc_keys[i] in row[3]:
                        xc_mtr_number.append(row[0])

        xc_mtr_values=[]
        for i in range(len(xc_mtr_number)):
            xc_mtr_value=[]
            for row in Eplusout:
                if len(row)>3:
                    if xc_mtr_number[i] == row[0]:
                        if self.xc_keys[i] not in row[3]:
                            xc_mtr_value.append(float(row[1]))
            xc_mtr_values.append(xc_mtr_value)

        return [yc_mtr_values, yc_var_values, xc_mtr_values]

    def datafield(self):
        iddfile = "./Energy+.idd"
        IDF.setiddname(iddfile)


        idfname = "Path + {}.idf".format(case_name)  # Put the path of your idf here
        idf = IDF(idfname)

        yc_df1 = pd.DataFrame(columns=self.yc_keys)
        vc_df1 = pd.DataFrame(columns=self.vc_keys)
        xc_df1 = pd.DataFrame(columns=self.xc_keys)  
        
        yc_df2 = pd.DataFrame(columns=self.yc_keys)
        vc_df2 = pd.DataFrame(columns=self.vc_keys)
        xc_df2 = pd.DataFrame(columns=self.xc_keys)  

        output_frequency=self.report_freq

        variable=[]
        for i in range(len(self.xc_keys)):
            variable1 = idf.newidfobject("Output:Variable".upper())
            variable1.Key_Value = '*'
            variable1.Variable_Name = self.xc_keys[i]
            variable1.Reporting_Frequency = output_frequency
            variable.append(variable1)
        idf.idfobjects['Output:Variable'.upper()]=variable

        meter=[]
        for i in range(len(self.yc_keys)):
            meter1 = idf.newidfobject("Output:Meter".upper())
            meter1.Key_Name = self.yc_keys[i]
            meter1.Reporting_Frequency = output_frequency
            meter.append(meter1)
        idf.idfobjects['Output:Meter'.upper()]=meter
        
        vc_meter=[]
        for i in range(len(self.vc_keys)):
            vc_meter1 = idf.newidfobject("Output:Variable".upper())
            vc_meter1.Key_Value = '*'
            vc_meter1.Variable_Name = self.vc_keys[i]
            vc_meter1.Reporting_Frequency = output_frequency
            vc_meter.append(vc_meter1)
        idf.idfobjects['Output:Variable'.upper()]=vc_meter + variable

        idf.idfobjects['RUNPERIOD'][0].Begin_Month=1
        idf.idfobjects['RUNPERIOD'][0].Begin_Day_of_Month=1
        idf.idfobjects['RUNPERIOD'][0].End_Month=12
        idf.idfobjects['RUNPERIOD'][0].End_Day_of_Month=31
        
        # Create a datafield path if it does not exist
        if not os.path.exists('Path of datafield'.format(t)):
            os.makedirs('Path of datafield'.format(t))

        if not os.path.exists('the parent folder'+'/res_folder_{}/datafield_2'.format(t)):
            os.makedirs('the parent folder'+'/res_folder_{}/datafield_2'.format(t))
        
        # Save the updated idf file
        idf.saveas('folder path' + 'Updated_Model_{}_randn_{}_clsp.idf'.format(t, self.case_name, self.report_freq))
        idfname1 = 'folder path' + '/Updated_Model_{}_randn_{}_clsp.idf'.format(t, self.case_name, self.report_freq)  
        epwfile = 'Path of epw'  # Put your epw file here
        subprocess.call(['Path of energyplus'+'energyplus.exe', '-d', 'Path of datafield folder', '-w', epwfile, idfname1])

        eso_file_1= 'Path of datafield folder'+ 'eplusout.eso'
        [ycoutput,vcoutput, xcoutput]=self.comp_data_reader(eso_file_1)


        # Create a dataframe for storing yc and vc
        yc_df1 = pd.DataFrame(data = np.array(ycoutput).T, columns=self.yc_keys)
        if len(self.vc_keys) == 0:   
            vc_df1 = pd.DataFrame(columns=self.zone_keys)              
        else:
            vc_df1 = pd.DataFrame(data = np.array(vcoutput).T, columns=self.zone_keys)

        # Create a dataframe for storing xc       
        xc_df1 = pd.DataFrame(data = np.array(xcoutput).T, columns=self.xc_keys)   


        epwfile_2 = 'the parent folder'+'/SPtMasterTable_52384_2012_amy.epw'
        subprocess.call(['D:/software_setup/EnergyPlusV9-1-0/energyplus.exe', '-d', 'the parent folder'+"/res_folder_{}/datafield_2'.format(t), '-w', epwfile_2, idfname1])

        eso_file_2='the parent folder'+'/res_folder_{}/datafield_2/eplusout.eso'.format(t)
        [ycoutput_2,vcoutput_2, xcoutput_2]=self.comp_data_reader(eso_file_2)

        yc_df2 = pd.DataFrame(data = np.array(ycoutput_2).T, columns=self.yc_keys)
        if len(self.vc_keys) == 0:
            vc_df2 = pd.DataFrame(columns=self.zone_keys)                       
        else:
            vc_df2 = pd.DataFrame(data = np.array(vcoutput_2).T, columns=self.zone_keys)
        
        xc_df2 = pd.DataFrame(data = np.array(xcoutput_2).T, columns=self.xc_keys)   

        yc_df = pd.concat([yc_df1,yc_df2],axis = 0)
        vc_df = pd.concat([vc_df1,vc_df2],axis = 0)
        xc_df = pd.concat([xc_df1,xc_df2],axis = 0)
        
        if len(self.vc_keys) == 1:
            vc_df['Zone Mean Air Temperature'] = np.nan
            total_area = 0
            for i in range(len(self.zone_keys)):
                total_area += self.conf['zone_areas'][i]
            total_area = round(total_area, 2)

            temp_series = vc_df['Zone Mean Air Temperature'].fillna(0)

            for n in range(len(self.zone_keys)):
                num = 'z{}'.format(n+1)
                temp_series += (vc_df[num]*self.conf['zone_areas'][n])/total_area

            vc_df['Zone Mean Air Temperature'] = temp_series

            vc_df = vc_df.drop(columns = self.zone_keys)   

        df = pd.concat([yc_df,vc_df, xc_df],axis = 1)

        df.to_csv('the path of datafield folder' + 'DATAFIELD_Multi_clsp_{}_{}.csv'.format(self.case_name, self.report_freq),index = False)
        df_single = pd.concat([yc_df.iloc[:,0],xc_df],axis = 1)
        df_single.to_csv('the path of datafield folder' + 'DATAFIELD_Single_clsp_{}_{}.csv'.format(self.case_name, self.report_freq),index = False)

    def datacomp(self):
        iddfile = "./Energy+.idd"
        IDF.setiddname(iddfile)

        yc_df_test = pd.DataFrame()
        yc_df = pd.DataFrame(columns = self.yc_keys)
        vc_df = pd.DataFrame(columns = self.vc_keys)
        xc_df = pd.DataFrame(columns = self.xc_keys)
        tc_df = pd.DataFrame(columns = self.tc_keys)
        
        for n in range(self.sm_num):
            idfname = 'the path of idf file folder' + 'RefBldgLargeOfficeNew2004_Chicago_{}.idf'.format(case_name)
            idf = IDF(idfname)

            output_frequency=self.report_freq

            variable=[]
            for i in range(len(self.xc_keys)):
                variable1 = idf.newidfobject("Output:Variable".upper())
                variable1.Key_Value = '*'
                variable1.Variable_Name = self.xc_keys[i]
                variable1.Reporting_Frequency = output_frequency
                variable.append(variable1)

            meter=[]
            for i in range(len(self.yc_keys)):
                meter1 = idf.newidfobject("Output:Meter".upper())
                meter1.Key_Name = self.yc_keys[i]
                meter1.Reporting_Frequency = output_frequency
                meter.append(meter1)
            idf.idfobjects['Output:Meter'.upper()]=meter

            vc_meter=[]
            for i in range(len(self.vc_keys)):
                vc_meter1 = idf.newidfobject("Output:Variable".upper())
                vc_meter1.Key_Value = '*'
                vc_meter1.Variable_Name = self.vc_keys[i]
                vc_meter1.Reporting_Frequency = output_frequency
                vc_meter.append(vc_meter1)
            idf.idfobjects['Output:Variable'.upper()] = vc_meter + variable

#             # 1. EPD
#             for i in range(len(idf.idfobjects['ELECTRICEQUIPMENT'])):
#                 idf.idfobjects['ELECTRICEQUIPMENT'][i].Watts_per_Zone_Floor_Area = self.LHS_result[n][0]

            # 2. Infiltration
            for i in range(len(idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE'])):
                idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE'][i].Flow_per_Exterior_Surface_Area = self.LHS_result[n][0]

#             # Issue
#             # 3. Cooling setpoint schedule, todo: could it be just cooling setpoint
#             for i in range(len(idf.idfobjects['SCHEDULE:COMPACT'])):
#                 if 'CLGSETP_SCH' in idf.idfobjects['SCHEDULE:COMPACT'][i].Name:
#                     idf.idfobjects['SCHEDULE:COMPACT'][i].Field_6=self.LHS_result[n][2]
#                     # idf.idfobjects['SCHEDULE:COMPACT'][i].Field_8=self.LHS_result[n][2]

            # 4. LPD
            for i in range(len(idf.idfobjects['LIGHTS'])):
                idf.idfobjects['LIGHTS'][i].Watts_per_Zone_Floor_Area = self.LHS_result[n][1]

            # Issue
            # 5. Heating setpoint schedule, todo: could it be just heating setpoint
            for i in range(len(idf.idfobjects['SCHEDULE:COMPACT'])):
                if 'HTGSETP_SCH' in idf.idfobjects['SCHEDULE:COMPACT'][i].Name:
                    idf.idfobjects['SCHEDULE:COMPACT'][i].Field_6 = self.LHS_result[n][2]
                    # idf.idfobjects['SCHEDULE:COMPACT'][i].Field_8=self.LHS_result[n][4]

            # 7.1 Component efficiency
            for i in range(len(idf.idfobjects['CHILLER:ELECTRIC:REFORMULATEDEIR'])):
                idf.idfobjects['CHILLER:ELECTRIC:REFORMULATEDEIR'][i].Reference_COP = self.LHS_result[n][3]

            # 7.2 Component efficiency
            for i in range(len(idf.idfobjects['BOILER:HOTWATER'])):
                idf.idfobjects['BOILER:HOTWATER'][i].Nominal_Thermal_Efficiency = self.LHS_result[n][4]

            # 8. Outdoor air
            for i in range(len(idf.idfobjects['DESIGNSPECIFICATION:OUTDOORAIR'])):
                idf.idfobjects['DESIGNSPECIFICATION:OUTDOORAIR'][i].Outdoor_Air_Flow_per_Person = self.LHS_result[n][5]

#             # 9. People
#             for i in range(len(idf.idfobjects['PEOPLE'])):
#                 idf.idfobjects['PEOPLE'][i].Zone_Floor_Area_per_Person=self.LHS_result[n][8]

#             # 10.1 Solar_Heat_Gain_Coefficient
#             for i in range(len(idf.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM'])):
#                 idf.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM'][i].Solar_Heat_Gain_Coefficient=self.LHS_result[n][9]

            # 10.2 U_Factor
            for i in range(len(idf.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM'])):
                idf.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM'][i].UFactor = self.LHS_result[n][6]

            # 10.3 Conductivity
            for i in range(len(idf.idfobjects['MATERIAL'])):
                idf.idfobjects['MATERIAL'][i].Conductivity = self.LHS_result[n][7]

            # Issue
            # 11. Equip schedule
            for i in range(len(idf.idfobjects['SCHEDULE:CONSTANT'])):
                if 'EQUIP_BASE' in idf.idfobjects['SCHEDULE:CONSTANT'][i].Name:
                    idf.idfobjects['SCHEDULE:CONSTANT'][i].Hourly_Value = self.LHS_result[n][8]


            idf.idfobjects['RUNPERIOD'][0].Begin_Month = 1
            idf.idfobjects['RUNPERIOD'][0].Begin_Day_of_Month = 1
            idf.idfobjects['RUNPERIOD'][0].End_Month = 12
            idf.idfobjects['RUNPERIOD'][0].End_Day_of_Month = 31

            if not os.path.exists('res_folder_{}'.format(t)):
                os.makedirs('the parent folder'+'/res_folder_{}'.format(t))

            if not os.path.exists('the parent folder'+'/res_folder_{}/res_{}_{}_test'.format(t, self.case_name, self.report_freq)):
                os.makedirs('the parent folder'+'/res_folder_{}/res_{}_{}_test'.format(t, self.case_name, self.report_freq))

            idf.saveas('the parent folder'+'/res_folder_{}/Updated_Model_{}_randn_{}_clsp.idf'.format(t, self.case_name, self.report_freq))
            # This IDF file is updated at each iteration.
            idfname1 = 'the parent folder'+'/res_folder_{}/Updated_Model_{}_randn_{}_clsp.idf'.format(t, self.case_name, self.report_freq)
            epwfile = 'the folder path of epw files' + 'SPtMasterTable_52384_2011_amy.epw'
            subprocess.call(['the path of energyplus'+'energyplus.exe', '-d', 
                             'the parent folder'+'/res_folder_{}/res_{}_{}_test'.format(t, self.case_name, self.report_freq), '-w', epwfile, idfname1])
            print('finished')

            eso_file='the parent folder'+'/res_folder_{}/res_{}_{}_test/eplusout.eso'.format(t, self.case_name, self.report_freq)
            [ycoutput,vcoutput, xcoutput]=self.comp_data_reader(eso_file)
 
            # Create dataframes for storing data            
            yc_cur = pd.DataFrame(data = np.array(ycoutput).T, columns = self.yc_keys)
            xc_cur = pd.DataFrame(data = np.array(xcoutput).T, columns = self.xc_keys)
            tc_cur = pd.DataFrame(np.reshape(list(self.LHS_result[n])*len(ycoutput[0]),(len(ycoutput[0]),len(self.tc_keys))),
                                  columns = self.tc_keys)

            if len(self.vc_keys) == 0:
                vc_cur = pd.DataFrame(columns = self.zone_keys)                    
            else:
                vc_cur = pd.DataFrame(data = np.array(vcoutput).T, columns = self.zone_keys)

            # Collect data
            yc_df = yc_df.append(yc_cur)

            vc_df = vc_df.append(vc_cur)

            xc_df = xc_df.append(xc_cur)

            tc_df = tc_df.append(tc_cur)


                
        # Loop through each zone
        if len(self.vc_keys) != 0:
            total_area = 0
            for i in range(len(self.zone_keys)):
                total_area += self.conf['zone_areas'][i]
            total_area = round(total_area, 2)

            temp_series = vc_df['Zone Mean Air Temperature'].fillna(0)

            for n in range(len(self.zone_keys)):
                num = 'z{}'.format(n+1)
                temp_series += (vc_df[num]*self.conf['zone_areas'][n])/total_area

            vc_df['Zone Mean Air Temperature'] = temp_series

            vc_df = vc_df.drop(columns=self.zone_keys)   

        # Concatenate data
        df = pd.concat([yc_df,vc_df,xc_df,tc_df],axis=1)

        proc_df = df
        proc_df_single = proc_df[[self.yc_keys[0]] + self.xc_keys+self.tc_keys]

        # Write results to csv for checking values
        proc_df.to_csv('the parent folder'+'/res_folder_{}/DATACOMP_Multi_{}_randn_{}_clsp.csv'.format(t, self.case_name, 
                 self.report_freq, self.case_name, self.report_freq),index = False)
        
        proc_df_single.to_csv('the parent folder'+'/res_folder_{}/DATACOMP_Single_{}_randn_{}_clsp.csv'.format(t, self.case_name, 
                 self.report_freq, self.case_name, self.report_freq),index = False)


def run_class(case_name, report_freq, simulation_num):
    conf = yaml.load(open("your_config_file_{}_{}.yaml".format(t, case_name)), Loader = yaml.FullLoader)
    # Put the list of containing lower and upper bounds of the selected calibration parameters here
    bounds = ['the range of the selected calibration parameters']  
    
    b_num = len(bounds)
    model = multi_data_simulator(case_name, conf, report_freq, simulation_num)
    model.LHSample(b_num,bounds,30)
    model.datacomp()

    model.datafield()

case_lst = ['Case name']  # Put your case name here
reso_lst = ['Monthly', 'Hourly']

case_name = case_lst[0]  # Set the case you want to run
report_freq = reso_lst[0]  # Set the report frequency
simulation_num = 30  # Set the number of running simulations
run_class(case_name, report_freq, simulation_num)

