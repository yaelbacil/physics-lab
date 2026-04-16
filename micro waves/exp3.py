import pandas as pd
import matplotlib.pyplot as plt

def get_experiment_data(df, start_col, end_col):
  # the function gets the relevant coloumns
  # returns specific sub_experiment data from the csv file 'data micro'
  exp_df = df.iloc[1:, start_col:end_col].copy()
  exp_df.columns = exp_df.iloc[0]
  exp_df = exp_df.iloc[1:].dropna(how='all').reset_index(drop=True)
  return exp_df


# calling data from github reposatory
# m1 - meusrment number 1
exp1 = "https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp1.csv"
exp3 = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%203%20mesure%201.csv'
exp4_m1 = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%204%20mesure%201.csv'
exp4_m2 = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%204%20mesure%202.csv'
micro_data_path = "https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/data%20micro.csv"

# creating dataframes
df_exp1 = pd.read_csv(exp1)
df_exp3 = pd.read_csv(exp3)
df_exp4_m1 = pd.read_csv(exp4_m1)
df_exp4_m2 = pd.read_csv(exp4_m2)
df_full = pd.read_csv(micro_data_path)

# calling subexperiments from 'data micro' file
df_exp2 = get_experiment_data(df_full, 4, 7)
df_exp5 = get_experiment_data(df_full, 8, 10)


# exp 3 data plotting
x_col = df_exp3.columns[0]
y_col = df_exp3.columns[1]

plt.figure(figsize=(8, 5))
plt.plot(df_exp3[x_col], df_exp3[y_col], marker='o', linestyle='-', color='b', label='data line')

plt.title(f'Amplitude vs Distance')
plt.xlabel(f'Distance [mm]')
plt.ylabel(f'Amplitude [mV]')

plt.grid(True)
plt.legend()
plt.show()