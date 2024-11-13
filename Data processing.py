
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
Battery_name = 'CS2_35'  
dir_path = 'dataset/'


def drop_outlier(array, count, bins):
    index = []
    range_ = np.arange(1, count, bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)

Battery = {}

print('Load the data directory structure ' + Battery_name + ' ...')
path = glob.glob(dir_path + Battery_name + '/*.xlsx')
dates = []
for p in path:
    df = pd.read_excel(p, sheet_name=1)
    print('Load file sequence ' + str(p) + ' ...')
    dates.append(df['Date_Time'][0])
idx = np.argsort(dates)
path_sorted = np.array(path)[idx]
print("The file structure was read successfully. There are {} files in total".format(len(path_sorted)))

count = 0
capacities = []
discharge_capacities = []
health_indicator = []
internal_resistance = []
CCCT = []
CVCT = []
Voltage = []
Current = []


discharge_time = []
k = []
RUL = []

for p in path_sorted:
    df = pd.read_excel(p, sheet_name=1)
    print('Load and Analytical data' + str(p) + ' ...')
    cycles = list(set(df['Cycle_Index']))

    for c in cycles:
        df_lim = df[df['Cycle_Index'] == c]
        # Charging
        df_c = df_lim[(df_lim['Step_Index'] == 2) | (df_lim['Step_Index'] == 4)]    # step 2 and step 4, charging
        c_v = df_c['Voltage(V)']
        c_c = df_c['Current(A)']
        c_t = df_c['Test_Time(s)']
        # CC or CV
        df_cc = df_lim[df_lim['Step_Index'] == 2]
        df_cv = df_lim[df_lim['Step_Index'] == 4]
        CCCT.append(np.max(df_cc['Test_Time(s)']) - np.min(df_cc['Test_Time(s)']))      # charge time
        CVCT.append(np.max(df_cv['Test_Time(s)']) - np.min(df_cv['Test_Time(s)']))      # charge time

        # Discharging
        df_d = df_lim[df_lim['Step_Index'] == 7]
        d_v = df_d['Voltage(V)']
        d_c = df_d['Current(A)']
        d_t = df_d['Test_Time(s)']
        d_im = df_d['Internal_Resistance(Ohm)']
        Voltage.append(np.average(d_v))
        Current.append(np.average(d_c))
        if len(list(d_c)) != 0:
            time_diff = np.diff(list(d_t))
            d_c = np.array(list(d_c))[1:]
            discharge_capacity = time_diff * d_c / 3600
            # print("discharge_capacity shape[0] is:{}".format(discharge_capacity.shape[0]))

            # discharge_capacity_sum = np.sum(discharge_capacity)
            # discharge_capacities.append(-1 * discharge_capacity_sum)
            discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])] 
            discharge_capacities.append(-1 * discharge_capacity[-1])
            discharge_time.append(sum(time_diff)) # RUL

            dec = np.abs(np.array(d_v) - 3.8)[1:]

            start = np.array(discharge_capacity)[np.argmin(dec)]
            dec = np.abs(np.array(d_v) - 3.4)[1:]
            end = np.array(discharge_capacity)[np.argmin(dec)]
            health_indicator.append((-1 * (end - start)))

            internal_resistance.append(np.mean(np.array(d_im)))
            
            
            count += 1
            
health_indicator = health_indicator/np.max(health_indicator)

discharge_capacities = np.array(discharge_capacities)
SOC = discharge_capacities/1.1
health_indicator = np.array(health_indicator)
internal_resistance = np.array(internal_resistance)
Current = np.array(Current)
Voltage = np.array(Voltage)

idx = drop_outlier(discharge_capacities, count, 40)
df_result = pd.DataFrame({'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),
                          'capacities': discharge_capacities[idx],
                          'Voltage': Voltage[idx],
                          'Current': Current[idx],
                          'resistance': internal_resistance[idx],
                          'SOC': SOC[idx],
                          'SOH': health_indicator[idx],     # SOH
                          })
Battery[Battery_name] = df_result
np.save(dir_path + Battery_name, Battery)
print("Data parsing succeeded. The .npy file was saved to {}".format(dir_path + Battery_name + '.npy'))

for p in path_sorted:
    df = pd.read_excel(p, sheet_name=1)
    cycles = list(set(df['Cycle_Index']))
    for c in cycles:
        df_lim = df[df['Cycle_Index'] == c]
        # Discharging
        df_d = df_lim[df_lim['Step_Index'] == 7]
        d_t = df_d['Test_Time(s)']
        discharge_time.append(d_t.max() - d_t.min())

print("Discharge durations for each cycle:", discharge_time)

plt.figure(figsize=(10, 6))
plt.plot(discharge_time, marker='o', linestyle='-', color='b')
plt.title('Cycle Duration Decay Curve')
plt.xlabel('Cycle Index')
plt.ylabel('Duration (s)')
plt.xticks(range(len(discharge_time)))
plt.grid()
plt.show()