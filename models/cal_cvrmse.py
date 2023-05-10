import numpy as np
import pandas as pd


# Function for calculating CVRMSE
def cal_cvrmse(y_pred, y_true):
	n_pred = np.shape(y_pred)[0]
	cvrmse = 100*np.sqrt(np.sum(np.square(y_pred-y_true))/(n_pred-1))/np.mean(y_true)

	return cvrmse 

# Function for computing the mean of each column
def col_means(y_pred_raw):
	df = y_pred_raw
	col_mean_df = df.mean(axis=0)

	return col_mean_df


name_lst = [case_name list]  # Put case names here
case_num = len(name_lst)
cvrmse_df = pd.DataFrame(columns=['CVRMSE'])  # Create a dataframe for storing CVRMSE results
parent_fd = "Pata of results"  # Put the path of results here

# Loop through each output of each case
for i, name in enumerate(name_lst):
    for j in range(case_num):
        conf = yaml.load(open(os.path.join(parent_fd, "conf_{}_{}.yaml".format(name_lst[i], j+1))), Loader=yaml.FullLoader)
        n_y = len(conf['vc_keys'])+len(conf['yc_keys'])
        
        for k in range(n_y):
            # Read the pred data of each output of each case
            y_pred_raw = pd.read_csv(os.path.join(os.getcwd(), '_case_{}_{}_y_pred{}.csv'.format(name_lst[i], j+1, k+1)), index_col=0)  
            y_pred_raw = y_pred_raw.T.reset_index(drop=True)

            # Read the DataField of each output of each case
            y_true = pd.read_csv(os.path.join(os.getcwd(), 'Path_of_Datafield + {}_{}'.format(name_lst[i], j+1)))

            # Calculate CVRMSE of each output of each case
            cvrmse = cal_cvrmse(col_means(y_pred_raw.T), y_true.iloc[:, k])
            cvrmse_df.loc['{}_{}_y_pred{}'.format(name_lst[i], j+1, k+1)] = cvrmse

# Create folder for storing the results
if not os.path.exists('./data'):
    os.makedirs('./data')
cvrmse_df.to_csv('./data/cvrmse_res.csv')