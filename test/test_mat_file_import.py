import scipy.io
import os
import pathlib
import scipy
import sys
import numpy as np
import scipy.io


path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
RESOURCE_FOLDER = path.joinpath("../../resources")


wind_speed_5 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-Wind-Data-Height-5.mat")
Conc_5 = scipy.io.loadmat("/home/ece213/CPSL-Sim_2/cuas/plume_data/Plume-C-Data-Height-5.mat")


#wind_speed_2 = scipy.io.loadmat(r"\cuas\plume_data\Plume-Wind-Data-Height-2.mat")
#Conc_2 = scipy.io.loadmat(r"\cuas\plume_data\Plume-C-Data-Height-2.mat")

print("Files imported!")

ws_Height_5 = wind_speed_5["ws"][-400000:, :]
rho_Height_5 = Conc_5 ["C"][-400000:, :]
print(f'{type(ws_Height_5)}')
print(f'Wind speed data shape:{ws_Height_5.shape}')
print(f'Concentration data shape:{rho_Height_5.shape}')
#print(ws_Height_5[0:10])

time_t = 1

idp = np.arange(100 * (time_t), 100 * (time_t + 1))
ws5 = ws_Height_5[idp]
rho5 = rho_Height_5[idp]


print(f'Wind speed data subset shape:{ws5.shape}')
print(f'Concentration data subset shape:{rho5.shape}')
'''


import numpy as np

# Generate a dataset with 1000 rows and 300 columns
rows = 1000
columns = 300

data = np.random.rand(rows, columns)

print("Generated dataset shape:", data.shape)
time_t = 1

idp = np.arange(100 * (time_t), 100 * (time_t + 1))
ws5 = data[idp]

print(f'{type(data)}')
print(f'Data subset shape:{data.shape}')
'''