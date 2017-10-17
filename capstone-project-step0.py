
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from scipy.optimize import minimize




# In[ ]:

data_list = ['wpi-assistments/math_2004_2005/ds92_tx_All_Data_172_2016_0504_081852.txt', \
             'wpi-assistments/math_2005_2006/ds120_tx_All_Data_265_2017_0414_065125.txt', \
             'wpi-assistments/math_2006_2007/ds339_tx_All_Data_1059_2015_0729_215742.txt']

for data_str in data_list:

    print("Preprocessing file", data_str)
    data = pd.read_csv(data_str, sep="\t", low_memory=False)
    data = data.dropna(axis=1, how='all')
    
    data['Day'] = data['Time'].apply(lambda x: x.split(" ")[0])
    data = data[['Anon Student Id', 'Session Id', 'Duration (sec)', 'Student Response Type', 'Problem Name', 'Problem View', 'Attempt At Step', 'Outcome', 'Day']]
    
    # Then, I collect the following information about each student (9 numerical parameters, in total):
    # - total number of sessions opened (`'Session Id'`);
    # - total number of problems entered (`'Problem Name'`);
    # - total number of attempts and hints made by the student (`'Student Response Type'`);
    # - fraction of correct attempts (`'Outcome'`, `'Student Response Type'`);
    # - total number of time spent for attempts and hints, respectively (`'Duration (sec)'`);
    # - problem complexity proxies: averaged maximal numbers of `'Problem View'` and `'Attempt At Step'` determined for each problem.
    
    stud_list = list(set(data['Anon Student Id']))
    
    
    # Taken from http://apmonitor.com/che263/index.php/Main/PythonDataRegression
    # and adopted for my purpose
    
    # Inplement C-stat (no need for binning), revelant formula is (5) in 
    # W.Cash paper http://adsabs.harvard.edu/doi/10.1086/156922  
    # see also B5 of https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XSappendixStatistics.html 
    
    # calculate y
    def calc_y(x):
        b = x[0]
        d = x[1]
        y = b*(xm)**(-d) # Fitting with powerlaw error function
        return y
    
    # define C-stat
    def C_stat(x):
        
    #    xm = np.array(attempts_data_stud_num['Attempt At Step'])
    #    ym = 1-np.array(attempts_data_stud_num['Outcome']) # 1-x because we fit the error rate
        # calculate y
        y = calc_y(x)
        # calculate C-stat
        Cstat = 0.0
        for i in range(len(ym)):
            Cstat += 2*(y[i] - ym[i]*np.log(y[i])) # C-stat, see eq.5 in http://adsabs.harvard.edu/doi/10.1086/156922 
        # return result
        return Cstat
    
    
    stud_data = pd.DataFrame()
    #stud_data.set_index = stud_list
    stud_data.reset_index()
    i = 0
    for stud_name in stud_list:
        stud_info_df = data[data['Anon Student Id'] == stud_name]
        
        # total number of sessions opened
        num_sessions = len(set(stud_info_df['Session Id']))
        
        # total number of problems entered
        num_problems = len(set(stud_info_df['Problem Name']))
        
        # total number of attempts made by the student 
        num_attempts = stud_info_df[stud_info_df['Student Response Type'] == 'ATTEMPT'].shape[0]
        
        # fraction of short attemps (with time <= 3 sec)
        if (num_attempts > 0):
            frac_3s_atts = stud_info_df[(stud_info_df['Student Response Type'] == 'ATTEMPT') & \
                                   (stud_info_df['Duration (sec)'].replace({'.': 0}).astype(float) <= 3.0)].shape[0] / num_attempts
        else:
            frac_3s_atts = 0
        
        # total number of hints made by the student 
        num_hints = stud_info_df[stud_info_df['Student Response Type'] == 'HINT_REQUEST'].shape[0]
        
        # fraction of short hints (with time <= 1 sec)
        if (num_hints > 0):
            frac_1s_hints = stud_info_df[(stud_info_df['Student Response Type'] == 'HINT_REQUEST') & \
                                   (stud_info_df['Duration (sec)'].replace({'.': 0}).astype(float) <= 1.0)].shape[0] / num_hints
        else:
            frac_1s_hints = 0
        
        # total number of days loading the system
        num_days = len(set(stud_info_df['Day']))
        # fraction of correct attempts
        if (num_attempts > 0):
            fraction_correct_attempts = stud_info_df[(stud_info_df['Student Response Type'] == 'ATTEMPT') & (stud_info_df['Outcome'] == 'CORRECT')].shape[0] / num_attempts
        else:
            fraction_correct_attempts = 0
        
        # total number of time spent for attempts (in seconds)
        total_time_attempts = stud_info_df[stud_info_df['Student Response Type'] == 'ATTEMPT']    ['Duration (sec)'].replace({'.': 0}).astype(float).sum()
        
        # total number of time spent for hints (in seconds)
        total_time_hints = stud_info_df[stud_info_df['Student Response Type'] == 'HINT_REQUEST']    ['Duration (sec)'].replace({'.': 0}).astype(float).sum()
        
        # averaged maximal numbers of 'Problem View'
        avg_max_problem_views = stud_info_df[['Problem Name', 'Problem View']].groupby(['Problem Name']).agg(np.max).mean()[0]
        avg_max_attempts_at_step = stud_info_df[['Problem Name', 'Attempt At Step']].groupby(['Problem Name']).agg(np.max).mean()[0]
    
        
        outcome_dict = {"CORRECT": 0, "INCORRECT": 1}
        stud_info_df = stud_info_df[stud_info_df['Student Response Type'] == 'ATTEMPT']
        #print(stud_info_df.head())
        stud_info_df_num = stud_info_df.replace(outcome_dict)
        stud_info_df_num.dropna(inplace=True)
        #print(stud_info_df_num.head())
        
        xm = np.array(stud_info_df_num['Attempt At Step'])
        ym = np.array(stud_info_df_num['Outcome']) # 1-x because we fit the error rate
    
        # initial guesses
        x0 = np.zeros(2)
        x0[0] = 0.7 # difficulty_parameter_b
        x0[1] = 0.5 # learning_rate_d
    
        # optimize
        # bounds on variables
        bounds_difficulty_parameter_b = (1e-3, 1.0e+1)
        bounds_learning_parameter_d = (-1.0e+2, 1.0e+2)
        solution = minimize(C_stat, x0, method='SLSQP', bounds=(bounds_difficulty_parameter_b, bounds_learning_parameter_d))
    
        # method = 'SLSQP' - original
        # other methods (L-BFGS-B, TNC) give the same results,
        # COBYLA is simply too slow ...
    
        x = solution.x
        y = calc_y(x)
    
        stud_name = i
        stud_data.loc[stud_name, 'learn_par_d'] = x[1]
        stud_data.loc[stud_name, 'diff_par_b'] = x[0]      
        stud_data.loc[stud_name, 'num_sess'] = num_sessions
        stud_data.loc[stud_name, 'num_days'] = num_days
        stud_data.loc[stud_name, 'num_probs'] = num_problems
        stud_data.loc[stud_name, 'num_atts'] = num_attempts
        stud_data.loc[stud_name, 'num_hints'] = num_hints
        stud_data.loc[stud_name, 'frac_corr_atts'] = fraction_correct_attempts
        stud_data.loc[stud_name, 'frac_3s_atts'] = frac_3s_atts
        stud_data.loc[stud_name, 'frac_1s_hints'] = frac_1s_hints
        stud_data.loc[stud_name, 'time_atts'] = total_time_attempts
        stud_data.loc[stud_name, 'time_hints'] = total_time_hints
        stud_data.loc[stud_name, 'probl_views'] = avg_max_problem_views
        stud_data.loc[stud_name, 'atts_at_step'] = avg_max_attempts_at_step
        try:
            stud_data.loc[stud_name, 'avg_time_att'] = total_time_attempts / num_attempts
        except ZeroDivisionError:
            stud_data.loc[stud_name, 'avg_time_att'] = 0
        try:
            stud_data.loc[stud_name, 'avg_time_hint'] = total_time_hints / num_hints
        except ZeroDivisionError:
            stud_data.loc[stud_name, 'avg_time_hint'] = 0 
        
        print("\r\t> Progress\t:{:.2%}".format((i + 1)/len(stud_list)), end='')
        i += 1
    
    
    savefile_str = data_str.split('/')[2].split('_')[0]
    
    with open(savefile_str+'.csv', "w") as f:
        stud_data.to_csv(f, header=True)
    
    print('\n')
    print('Finished, saved to', savefile_str+'.csv')

