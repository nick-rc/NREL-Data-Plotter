'''
Nicholas Craig
03/22/18
    This program is designed to process solar irradience data from the NREL
    database and produce a mapping of various ROI, $/Watt values for the
    United States
'''
# Imports
import profile
import math, os
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import json

import glob # For directory and pathname processing

# DYNAMIC DICT CLASS
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def get_data_dirs():
    # Start by globbing the data directory for the NREL Data folders
    data_folders = glob.glob('data/*')
    # Returns an array of all the directories
    return data_folders

# Step 1 - Get a map of all coordinates and Location IDs
def get_location_dict():
    # Create a dictionary of files by getting the:
        # Location ID
        # Lat
        # Long
    location_dict = {}
    # Loop through all of the directories
    for dir in get_data_dirs():
        # Get all the get_data_filenames
        data_files = glob.glob(dir+'/*')
        # Loop through all the data_files
        for fn in data_files:
            # Split to just get the filename
            fn = fn.split('/')[-1]
            # Check if filename is in filename_list
            # print(fn.split('_')[0])
            # print(location_dict.keys())
            if fn.split('_')[0] in location_dict.keys():
                # Dont add again
                #print("Already Found")
                pass
            else:
                # Split for adding to other dicts
                split_data = fn.split('_')
                # Add fn to filename_list
                new_loc = {'long':split_data[1], 'lat':split_data[2]}
                location_dict[int(split_data[0])] = new_loc
    # After adding all locations, return the location dict.
    # plot_locations(location_dict)
    return location_dict

# Plot all the locations in our map.
def plot_locations(loc_dict):
	# Use plt to create a plot of locations in the dict.
	# First, sort the dictionary items into a list
	location_list = sorted(loc_dict.items())
	# Then, split the lat and long values
	ids, coords = zip(*location_list)
	# Create a list of longs and lats key:vals from tuple
	longs = []
	lats = []
	for coord in coords:
		# Append Longs
		longs.append(eval(coord['long']))
		lats.append(eval(coord['lat']))
	# Creates the plot
	plt.scatter(lats, longs)
	plt.show()
	return location_list


# Need to process all files and return an average value for each nm value
def get_dataframes(loc_dict):
	# Start by looping through the data directories and files
	for dir in get_data_dirs():
		print("Processing New Directory")
		# Get all datafile paths
		data_files = glob.glob(dir+'/*')
		# Loop through each file
		for file in data_files:
			fp = open(file)
			id_line = ''
			for i, line in enumerate(fp):
				if i == 1:
					# Set the line with Location ID to a variable
					id_line = line
					# Close the IO
					fp.close()
					break
				elif line == 2:
					break
				else:
					pass
					# Splits line and sets the location ID
			loc_ID = id_line.split(',')[1]
			# Creates a dataframe of the current file
			file_df = pd.read_csv(file, sep=',', header=2, na_values='0')
			# file_df.fillna('0')
			# Add the mean values for each series to the loc_id dictionary
			loc_dict[eval(loc_ID)]['avg_temp'] = file_df["hr mean ambient tmp"].mean()
			loc_dict[eval(loc_ID)]['avg_precip'] = file_df["Precip water (cm)"].mean()
			for col in list(file_df):
				# Check ot see if col header has DNI in it
				if "DNI" in col:
					# DNI Column - Add value and average
					loc_dict[eval(loc_ID)][col] = file_df[col].mean()
				else:
					pass
			fp.close()
			print("*",end='', flush=True)
	return loc_dict

# Function to sum the DNI values of the dict and return a dict with:
# 	LOC ID - temp, pressure, lat, long, DNI Sum
def get_summed_DNI(loc_dict):
	# Create new simple dict
	sum_dict = {}
	# First - Loop through all the LOC IDs in the dictionary
	for key, data_pts in loc_dict.items():
		# Create a variable to hold the DNI sum
		sum_dict[key] = {}
		dni_sum = 0
		# Then loop through using itervalues
		for data_key, value in data_pts.items():
			# Check if key contains DNI, if it does, sum the value
			if 'DNI' in data_key:
				# Key is a DNI value - add value to DNI sum
				dni_sum += data_pts[data_key]
			else:
				# Add value to sum_dict
				sum_dict[key][data_key] = value
		# Add DNI_sum to loc_dict
		sum_dict[key]['DNI Sum'] = dni_sum
	return sum_dict

# Create a function to cache the dictionary
def cache_dict(dict):
	# Create a json object from the dict
	json_dict = json.dumps(dict)
	# Cache the DB in a file
	f = open("dict_cache.json","w")
	f.write(json_dict)
	f.close()
	return None

# Create a function to pull the cache
def uncache_dict():
	f = open("dict_cache.json","r")
	json_str = f.read()
	json_data = json.loads(json_str)
	return json_data

# Manipulate DNI to get other values.
def split_DNI_Dict(dict):
	# First get a list of locations that are sorted.
	loc_list = sorted(dict.items())
	# Get list of location Ids and
	ids, loc_data = zip(*loc_list)
	temp_list = []
	# Create 5 Lists - Temp, Lat, Long, Precip, DNI Sum
	temp_list = [d["avg_temp"] for d in loc_data]
	precip_list = [d["avg_precip"] for d in loc_data]
	lat_list  = [float(eval(d["lat"])) for d in loc_data]
	long_list = [float(eval(d["long"])) for d in loc_data]
	sumDNI_list = [float(d["DNI Sum"]) for d in loc_data]

	# Return the 5 lists
	# Type
	return temp_list, precip_list, lat_list, long_list, sumDNI_list

	# Create a histogram(heatmap) plotter here

# Create Functions to manipulate the DNI Data
def get_irradience(dni_list):
	# COnvert DNI to W/m2 irradience
	avg_dni = sum(dni_list)/float(len(dni_list))

	# Scale the entire list
	scaled_DNI = [x*24 for x in dni_list]
	avg_irrad = sum(scaled_DNI)/float(len(scaled_DNI))
	print("Average irradience value is: ", avg_irrad, "W/m2")

	return scaled_DNI


def threeD_plotter(lat_list, long_list, z_list_in):
	# Create figure to add plots to
	# Figure size is 18 x 9 inches
	fig = plt.figure(figsize=(24,12))
	# Add the z_list1 subplot
	ax1  = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel("Latitude")
	ax1.set_ylabel("Longitude")
	ax1.set_zlabel("DNI Values")
	# Numpy methods
	# lat_list, long_List = np.meshgrid(lat_list, long_list)
	print(len(lat_list))
	print(len(long_list))
	print(len(z_list_in))
	# lat_list, long_list = np.meshgrid(lat_list, long_list, z_list1)
	# np.array(lat_list)
	# np.array(long_list)
	# Create a numpy array of all the lists
	lt_list = np.array(lat_list)
	lg_list = np.array(long_list)
	z_list = np.array(z_list_in)
	# Create surface plot
	# ax.plot_surface(lt_list, lg_list, z_list, , linewidth=1, antialiased=False)
	sc1 = ax1.scatter(lt_list, lg_list, z_list, zdir='z', marker=".", c=z_list,  cmap=cm.seismic, s=50) # 'viridis'
	cb = fig.colorbar(sc1)
	# surface2 = ax.plot_surface(lat_list, long_list, z_list2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	# surface2 = ax.plot_surface(lat_list, long_list, z_list3, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# Customize the z axis.
	# ax.set_zlim(0, 50.01)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# Second test subplot
	'''
	ax2 = fig.add_subplot(122, projection='3d')
	dx = np.ones(len(lat_list))/2
	dy = np.ones(len(long_list))/2
	z_pos = np.zeros(len(z_list))
	norm = plt.Normalize(z_list.min(), z_list.max())
	colors = cm.BrBG(norm(z_list))

	# ax2.bar3d(lt_list, lg_list, z_pos, dx, dy, z_list, alpha=0.5, color=colors)
	'''
	# Add a color bar which maps values to colors.
	# fig.colorbar(s1, shrink=0.5, aspect=5)
	# ax.plot3D(lat_list, long_list, z_list)
	plt.show()
	return fig

# Main Function
def main():
    # Do stuff in Main
    run_type = input("What would you like to run?('Get Data'/'Plot Data')")
    print(run_type)
    if run_type == 'Get Data':
        # Pull the NREL Data and cache it
        summed_dict = get_summed_DNI(get_dataframes(get_location_dict()))
        # cache_dict(summed_dict)
        cache_dict(summed_dict)
    elif run_type == 'Plot Data':
        # Pull the cached data
        saved_data = uncache_dict()
        temp_l, precip_l, lat_l, long_l, sum_l = split_DNI_Dict(saved_data)
        # Scale DNI list
        dni_list = get_irradience(sum_l)
        fig = threeD_plotter(lat_l, long_l, dni_list)
    else:
        # Bad input
        print("Bad Entry")

# plot_locations(get_location_dict())
# summed_dict = get_summed_DNI(get_dataframes(get_location_dict()))
    print("Fin")
    return None


# Only run main if specifically called
if __name__ == '__main__':
    print("Running system Tests...\n")
    profile.run('main()')
else:
    print("Imported parse_data.py for processing.\n")
