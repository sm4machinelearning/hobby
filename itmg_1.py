e# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
192837
np.random.seed(10)

'''
pi = np.pi
nums = 101
x = np.linspace(-pi, pi, nums)
y = np.sin(x)*2
noise1 = np.random.rand(nums)
noise2 = np.random.rand(nums)
hhhy1 = y + noise1
y2 = y + noise2
plt.scatter(x, y1)
plt.scatter(x, y2)
plt.show()

var1 = np.var(y1)
var2 = np.var(y2)

print (var1, var2)
'''

y1 = np.random.normal(0, 2, 5000)
y2 = np.random.normal(0, 1, 5000)

print (np.var(y1))
print (np.var(y2))

plt.hist(y1, 100)
plt.hist(y2, 100)
plt.show()


def plot_lcr_roc(y_true, y_pred, save_path, save_name):
    x_model, y_model = get_lcr_coordinates(y_true, y_pred)
    x_perf, y_perf = get_lcr_coordinates(y_true, y_true)

    plt.style.use('bmh')
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    # # Plot perfect model curve
    ax.plot(x_perf, y_perf, 
            color='#f3be00', 
            label='Perfect Model (area = ' + str(np.round(auc(x_perf, y_perf), 4)) + ')')
    # Plot model curve
    ax.plot(x_model, y_model, 
            color='#009286',
            label='Model (area = ' + str(np.round(auc(x_model, y_model), 4)) + ')')
    # Plot random guess model line
    ax.plot(np.linspace(min(x_model), max(x_model)), 
            np.linspace(min(x_model), max(x_model)),
            color='black', 
            linestyle='--', label='Random guess model (area = 0.5)')
    plt.legend(loc='lower right')
    plt.title('Loss Capture Ratio ROC - ' + data_name)
    if path_save:
        plt.savefig(path_save + 'Loss Capture Ratio ROC - ' + data_name + '.png')
    plt.show()

    # Plot the LCR curve
if plot_curve:
    plt.figure()
    plt.plot(model_lcr_x, model_lcr_y, label='Model')
    plt.plot(real_lcr_x, real_lcr_y, label='Perfect model')
    plt.plot(np.array([0,1]), np.array([0,1]), label='Random model')
    plt.legend()
    plt.show()

return lcr_,  pvalue, CI 



    def loss_capture_ratio_function_numweight(self, realized, predicted, weight, save_path=None, save_name=None):
        
        def trapez_integral(x,y):
            bases = x[1:] - x[:-1]
            area = np.dot(bases, y[1:]+y[:-1])/2
            return area
        
        def lcr_curve(realized, predicted):
            """
            This subfunction builds the LCR curve given the rates (LGDs or loss rates) and returns x and y
            """
            # Put in a dataframe to perform aggregations
            rates_df = pd.DataFrame({'realized':realized, 'predicted':predicted}) 
            # Aggregate values by summing and counting
            rates_df_aggr = rates_df.groupby(by='predicted')['realized'].agg(['sum', 'count'])
            # Sort by decreasing predicted value
            rates_df_aggr.sort_index(ascending=False, inplace=True)
            # Compute x, counts of realized values
            x_coordinates = (rates_df_aggr['count'].cumsum()/rates_df_aggr['count'].sum()).values 
            # Compute y, cumulative share of realized value
            y_coordinates = (rates_df_aggr['sum'].cumsum()/rates_df_aggr['sum'].sum()).values 
            # Add the origin to the curve
            x_coordinates = np.concatenate(([0], x_coordinates)) 
            # Add the origin to the curve
            y_coordinates = np.concatenate(([0], y_coordinates)) 
            return x_coordinates, y_coordinates
        
        
        def plot_lcr_roc(model_lcr_x, model_lcr_y, real_lcr_x, real_lcr_y, save_path, save_name):
            x_perf, y_perf = real_lcr_x, real_lcr_y
            x_model, y_model = model_lcr_x, model_lcr_y
        
            plt.style.use('bmh')
            fig, ax = plt.subplots()
            ax.set_facecolor('white')
            
            # # Plot perfect model curve
            ax.plot(x_perf, y_perf, 
                    color='#f3be00', 
                    label='Perfect Model (area = ' + str(np.round(trapez_integral(x_perf, y_perf), 4)) + ')')
            
            # Plot model curve
            ax.plot(x_model, y_model, 
                    color='#009286',
                    label='Model (area = ' + str(np.round(trapez_integral(x_model, y_model), 4)) + ')')
            
            # Plot random guess model line
            ax.plot(np.linspace(min(x_model), max(x_model)), 
                    np.linspace(min(x_model), max(x_model)),
                    color='black', 
                    linestyle='--', label='Random guess model (area = 0.5)')
            
            plt.legend(loc='lower right')
            if save_path:
                plt.savefig(os.path.join(save_path, save_name + '_numwt_LCR_curve.png'))
            plt.show()
        
        model_lcr_x, model_lcr_y = lcr_curve(realized, predicted)
        real_lcr_x, real_lcr_y = lcr_curve(realized, realized)
        area_model = trapez_integral(model_lcr_x, model_lcr_y)
        area_perfect_model = trapez_integral(real_lcr_x, real_lcr_y)
        lcr_ = (area_model-0.5)/(area_perfect_model-0.5)
        
        if save_path:
            plot_lcr_roc(model_lcr_x, model_lcr_y, real_lcr_x, real_lcr_y, save_path, save_name)
        
        return lcr_
    
    





    def loss_capture_ratio_function_expweight(self, realized, predicted, weight, save_path=None, save_name=None):
    
        def trapez_integral(x,y):
            bases = x[1:] - x[:-1]
            area = np.dot(bases, y[1:]+y[:-1])/2
            return area
        
        def lcr_curve(realized, predicted, raw):
            """
            This subfunction builds the LCR curve given the rates (LGDs or loss rates) and returns x and y
            """
            # Put in a dataframe to perform aggregations
            rates_df = pd.DataFrame({'realized':realized, 'predicted':predicted, 'raw':raw})
            # Aggregate values by summing and counting
            rates_df_aggr = rates_df.groupby(by='predicted')['realized', 'raw'].agg(['sum', 'count'])
            # set index as the sum of raw
            rates_df_aggr.set_index(('raw', 'sum'))
            # Sort by decreasing predicted value
            rates_df_aggr.sort_index(ascending=False, inplace=True)
            # Compute x, counts of realized values
            x_coordinates = (rates_df_aggr['realized', 'count'].cumsum()/rates_df_aggr['realized', 'count'].sum()).values 
            # Compute y, cumulative share of realized value
            y_coordinates = (rates_df_aggr['realized', 'sum'].cumsum()/rates_df_aggr['realized', 'sum'].sum()).values 
            # Add the origin to the curve
            x_coordinates = np.concatenate(([0], x_coordinates)) 
            # Add the origin to the curve
            y_coordinates = np.concatenate(([0], y_coordinates)) 
            return x_coordinates, y_coordinates
        
        
        def plot_lcr_roc(model_lcr_x, model_lcr_y, real_lcr_x, real_lcr_y, save_path, save_name):
            x_perf, y_perf = real_lcr_x, real_lcr_y
            x_model, y_model = model_lcr_x, model_lcr_y
        
            plt.style.use('bmh')
            fig, ax = plt.subplots()
            ax.set_facecolor('white')
            
            # # Plot perfect model curve
            ax.plot(x_perf, y_perf, 
                    color='#f3be00', 
                    label='Perfect Model (area = ' + str(np.round(trapez_integral(x_perf, y_perf), 4)) + ')')
            
            # Plot model curve
            ax.plot(x_model, y_model, 
                    color='#009286',
                    label='Model (area = ' + str(np.round(trapez_integral(x_model, y_model), 4)) + ')')
            
            # Plot random guess model line
            ax.plot(np.linspace(min(x_model), max(x_model)), 
                    np.linspace(min(x_model), max(x_model)),
                    color='black', 
                    linestyle='--', label='Random guess model (area = 0.5)')
            
            plt.legend(loc='lower right')
            if save_path:
                plt.savefig(os.path.join(save_path, save_name + '_expwt_LCR_curve.png'))
            plt.show()
        
        predicted = predicted * weight
        realized = realized * weight
        model_lcr_x, model_lcr_y = lcr_curve(realized, predicted, predicted)
        real_lcr_x, real_lcr_y = lcr_curve(realized, realized, realized)
        area_model = trapez_integral(model_lcr_x, model_lcr_y)
        area_perfect_model = trapez_integral(real_lcr_x, real_lcr_y)
        lcr_ = (area_model-0.5)/(area_perfect_model-0.5)
        
        if save_path:
            plot_lcr_roc(model_lcr_x, model_lcr_y, real_lcr_x, real_lcr_y, save_path, save_name)
        
        return lcr_








        bootstrapexp = 0
        if bootstrapexp == 1:
            lcr_exp = self.loss_capture_ratio_function(realized, predicted, weight)
            range_exp = []
            data = np.asarray((realized, predicted, weight)).T        
            for i in range(n_boots):
                booted_data = self.bootstrapped(data)
                realized_boot, predicted_boot, weight_boot = booted_data[:, 0], booted_data[:, 1], booted_data[:, 2]
                range_exp.append(self.loss_capture_ratio_function(realized_boot, predicted_boot, weight_boot))
            return lcr, min(range_lcr), max(range_lcr), lcr_exp, min(range_exp), max(range_exp)
        else:




        range_lcr = []
        data = np.asarray((realized, predicted, weight)).T
        for i in range(n_boots):
            booted_data = self.bootstrapped(data)
            realized_boot, predicted_boot = booted_data[:, 0], booted_data[:, 1]
            weighted_boot = booted_data[:, 2]
            range_lcr.append(
                self.loss_capture_ratio_function(realized_boot, predicted_boot, np.ones(len(weight)))
            )










# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:05:06 2020

@author: C64990
"""

def detect_outliers_mad(data, var, mad_thresh=3.5, mad_constant=1.4826):
    """
    Detects outliers based on the mean absolute deviation (MAD, modified Z-score)
    :param data: pandas Series or Series like
        The values for which outliers would be detected
    :param mad_thresh: float
        The threshold with which the MAD-stat would be compared. MV use 3.5
    :param mad_constant: float
        The constant used in the MAD-stat computation. MV use 0.6745
    :return: outliers_mad: pandas DataFrame
       A dataframe containing the name of the chosen variable (var), the name of the stastical test,
        the number of total, lower and upper  outliers and the boundaries.
        The dataframe has only 1 row
    
    https://docs.oracle.com/cd/E17236_01/epm.1112/cb_statistical/frameset.htm?ch07s02s10s01.html
    https://core.ac.uk/download/pdf/206095228.pdf
    """
    
    data = data.values
    med = np.median(data)
    med_abs_dev = np.median(np.abs(data - np.median(data)))    
    mad_value = med_abs_dev * mad_constant * mad_thresh
    lower_b, upper_b = med - mad_value, med + mad_value
    
    df_out = pd.DataFrame({'values': data, 'mad_score': mad_value})
    df_out['is_outlier'] = 0 
    outlier_indices = np.where((data > upper_b) & (data < lower_b))
    df_out['is_outlier'].iloc[outlier_indices] = 1
    
    number_out = dict()
    number_low_out = dict()
    number_up_out = dict()

    number_out[var] = df_out['is_outlier'].sum()
    number_low_out[var] = np.where(data > (med + mad_value), 1, 0).sum()
    number_up_out[var] = np.where(data < (med - mad_value), 1, 0).sum()

    outliers_mad = pd.DataFrame({'variable': var,
                                 'method': 'modified Z-score',
                                 'nmb_outliers': list(number_out.values()),
                                 'Lower_Boundary': lower_b,
                                 'Upper_Boundary': upper_b,
                                 'nmb_low_outliers': list(number_low_out.values()),
                                 'nmb_up_outliers': list(number_up_out.values())})
    return outliers_mad, df_out['is_outlier'].values



def detect_outliers_mad(data, var, mad_thresh=3.5, mad_constant=0.6745):
    """
    Detects outliers based on the mean absolute deviation (MAD, modified Z-score)
    :param data: pandas Series or Series like
        The values for which outliers would be detected
    :param mad_thresh: float
        The threshold with which the MAD-stat would be compared. MV use 3.5
    :param mad_constant: float
        The constant used in the MAD-stat computation. MV use 0.6745
    :return: outliers_mad: pandas DataFrame
       A dataframe containing the name of the chosen variable (var), the name of the stastical test,
        the number of total, lower and upper  outliers and the boundaries.
        The dataframe has only 1 row
    """
    data = data.values
    med_abs_dev = np.median(np.abs(data - np.median(data)))
    mod_z = mad_constant * ((data - np.median(data)) / med_abs_dev)
    df_out = pd.DataFrame({'values': data, 'mad_score': mod_z})
    df_out['is_outlier'] = np.where(abs(stats.zscore(data)) > mad_thresh, 1, 0)

    number_out = dict()
    number_low_out = dict()
    number_up_out = dict()
    number_out[var] = df_out['is_outlier'].sum()
    number_low_out[var] = np.where(stats.zscore(data) < -mad_thresh, 1, 0).sum()
    number_up_out[var] = np.where(stats.zscore(data) > mad_thresh, 1, 0).sum()

    outliers_mad = pd.DataFrame({'variable': var,
                                 'method': 'modified Z-score',
                                 'nmb_outliers': list(number_out.values()),
                                 'Lower_Boundary': np.median(data) - mad_thresh * med_abs_dev / mad_constant,
                                 'Upper_Boundary': np.median(data) + mad_thresh * med_abs_dev / mad_constant,
                                 'nmb_low_outliers': list(number_low_out.values()),
                                 'nmb_up_outliers': list(number_up_out.values())})
    return outliers_mad, df_out['is_outlier'].values








def detect_outliers_mad(data, var, mad_thresh=3.5, mad_constant=0.6745):#1.4826):
    """
    Detects outliers based on the mean absolute deviation (MAD, modified Z-score)
    :param data: pandas Series or Series like
        The values for which outliers would be detected
    :param mad_thresh: float
        The threshold with which the MAD-stat would be compared. MV use 3.5
    :param mad_constant: float
        The constant used in the MAD-stat computation. MV use 0.6745
        value of 1.4826 is found in the article
        https://core.ac.uk/download/pdf/206095228.pdf    
    :return: outliers_mad: pandas DataFrame
       A dataframe containing the name of the chosen variable (var), the name of the stastical test,
        the number of total, lower and upper  outliers and the boundaries.
        The dataframe has only 1 row
    
    https://docs.oracle.com/cd/E17236_01/epm.1112/cb_statistical/frameset.htm?ch07s02s10s01.html
    """
    
    data = data.values
    med = np.median(data)
    med_abs_dev = np.median(np.abs(data - np.median(data)))
    #mad_value = med_abs_dev * mad_constant * mad_thresh
    mad_value = mad_constant * ((data - np.median(data)) / med_abs_dev)
    lower_b, upper_b = med - mad_value, med + mad_value
    
    df_out = pd.DataFrame({'values': data, 'mad_score': mad_value})
    df_out['is_outlier'] = 0 
    outlier_indices = np.where((data > upper_b) | (data < lower_b))
    df_out['is_outlier'].iloc[outlier_indices] = 1

    number_out = dict()
    number_low_out = dict()
    number_up_out = dict()

    number_out[var] = df_out['is_outlier'].sum()
    number_low_out[var] = np.where(data < (lower_b), 1, 0).sum()
    number_up_out[var] = np.where(data > (upper_b), 1, 0).sum()

    outliers_mad = pd.DataFrame({'variable': var,
                                 'method': 'modified Z-score',
                                 'nmb_outliers': list(number_out.values()),
                                 'Lower_Boundary': lower_b,
                                 'Upper_Boundary': upper_b,
                                 'nmb_low_outliers': list(number_low_out.values()),
                                 'nmb_up_outliers': list(number_up_out.values())})
    return outliers_mad, df_out['is_outlier'].values





