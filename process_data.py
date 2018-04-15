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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
	# print("Average DNI value is: ", avg_dni, "W/m2")

	# Scale the entire list
	scaled_DNI = [x*24 for x in dni_list]
	avg_irrad = sum(scaled_DNI)/float(len(scaled_DNI))
	print("Average irradience value is: ", avg_irrad, "W/m2")
	return scaled_DNI

# Get power output
def get_powerout(irrad_list, efficiency):
    # Set efficiency
    # efficiency = 10 # %
    panel_power = [x*efficiency/100 for x in irrad_list]
    avg_power = sum(panel_power)/float(len(panel_power))
    print("Average Panel output is: ", avg_power, "W/m2")
    return panel_power

# Get price per watt lists - for TWS and for buyers
def get_ModuleCosts(pwr_list):
    # Set cost to produce m2
    prod_cost = 23 # $/m2
    sell_cost = 50 # $/m2
    prod_ppw = [prod_cost/x for x in pwr_list]
    avg_prod_ppw = sum(prod_ppw)/float(len(prod_ppw))
    sell_ppw = [sell_cost/x for x in pwr_list]
    avg_sell_ppw = sum(sell_ppw)/float(len(sell_ppw))
    print("Average Prod PPW is: ", avg_prod_ppw, "$/W")
    print("Average Sell PPW is: ", avg_sell_ppw, "$/W")
    return prod_ppw, sell_ppw

def calc_LCOE(cost, n):
	i     = 0.05 # Discount rate of capital(10%)
	cf    = 0.2
	OandM = .001
	n = float(n)
	crf   = (i*(i+1)**n)/((i+1)**n - 1)
	LCOE_val = (cost * 10**6 * crf)/(cf*8760) + OandM
	return LCOE_val

def get_avg(in_list):
	# Return the average value.
	return sum(in_list)/float(len(in_list))


def get_LCOE(pwr_list, cost_list, lifetime):
    # Set the standard values for the solar cell
    # mwh_per_lifetime = [x*10/1000 for x in pwr_list] # *8.765813
    life_cost = [x*4 for x in cost_list]
    lcoe_list = [calc_LCOE(x, lifetime) for x in life_cost]
    # print(lcoe_list)  # [c/p for c, p in zip(life_cost, mwh_per_lifetime)]
    avg_lcoe = sum(lcoe_list)/float(len(lcoe_list))
    # print("Average LCOE is: ", avg_lcoe, "$/MWh")
    return lcoe_list

def plot_LCOE_vals(efficiencies, lifetimes, sum_l):
    # Create a figure
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.set_title("Levelized Cost of Energy \n Comparison Chart", color='white', fontsize=18)
    ax.set_xlabel("Lifetime (years)", color='white', fontsize=18)
    ax.set_ylabel("LCOE ($/MWh)", color='white', fontsize=18)
    lcoe_legend = []
    ax.tick_params('both', color='white', labelcolor='white',labelsize=16, size=5)
    # For each efficiency, create a list of LCOE values
    lcoe_compares = []
    lcoe_avg_list = []
    for efficiency in efficiencies:
        # loop through lifetimes and colculate LCOE
        for lifetime in lifetimes:
            irr = get_irradience(sum_l)
            pwr = get_powerout(irr, efficiency)
            p_cost, s_cost = get_ModuleCosts(pwr)
            lcoe_l = get_LCOE(pwr, s_cost, lifetime)
            # print(get_avg(lcoe_l))
            lcoe_avg_list.append(get_avg(lcoe_l))
        # Once the lifetimes list is full, append it to the ocmpare list
        lcoe_compares.append(lcoe_avg_list)
        lcoe_legend.append("Efficiency = {:.1f}".format(efficiency))
        plt.plot(lifetimes, lcoe_avg_list, marker='.', markersize=10, linewidth=3)
        plt.grid(True, alpha=0.7, linestyle='-', linewidth=2, color='white')# , color='white')
        lcoe_avg_list = []
    # Return the list of lists
    plt.legend(lcoe_legend, loc='upper right')
    plt.show()
    fig.savefig('test-2.png', transparent=True)
    return lcoe_compares

def threeD_plotter(lat_list, long_list, z_list_in, z_label):
	# Create figure to add plots to
	# Figure size is 18 x 9 inches
	fig = plt.figure(figsize=(24,12))
	# Add the z_list1 subplot
	ax1  = fig.add_subplot(111, projection='3d')
	ax1.set_xlabel("Latitude")
	ax1.set_ylabel("Longitude")
	ax1.set_zlabel(z_label)
	ax1.xaxis._axinfo['label']['space_factor'] = 10
	ax1.yaxis._axinfo['label']['space_factor'] = 10
	ax1.zaxis._axinfo['label']['space_factor'] = 10
	ax1.set_title("Levelized Cost of Energy \n Data for the United States")
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
	sc1 = ax1.scatter(lt_list, lg_list, z_list, zdir='z', marker=".", c=z_list,  cmap=cm.YlGn_r, s=50) # 'viridis'
	ax1.view_init(elev=75, azim=-80)
	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	# divider = make_axes_locatable(ax1)
	# cax = divider.append_axes('top', size="5%", pad=0.05)
	cb = fig.colorbar(sc1, ax=ax1)
	cb.set_label('$USD/MWh')
	# Add a color bar which maps values to colors.
	# fig.colorbar(s1, shrink=0.5, aspect=5)
	# ax.plot3D(lat_list, long_list, z_list)
	fig.savefig('3D-Plot.png')
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
        pwr_list = get_powerout(dni_list, 10)
        prod_cost, sell_cost = get_ModuleCosts(pwr_list)
        lcoe_list = get_LCOE(pwr_list, sell_cost, 10)
        fig = threeD_plotter(lat_l, long_l, lcoe_list, 'LCOE ($USD/MWh)')
    elif run_type == 'LCOE Data':
        saved_data = uncache_dict()
        temp_l, precip_l, lat_l, long_l, sum_l = split_DNI_Dict(saved_data)
        # Create Efficiency Range
        effs = [x/2 for x in range(20, 45, 5)]
        print(effs)
        # Create Lifetimes range
        lifes = range(5, 36, 1)
        # Plot the LCOE List
        lcoe_comp = plot_LCOE_vals(effs, lifes, sum_l)
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
