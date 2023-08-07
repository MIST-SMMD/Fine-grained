''' Calculation of sampling points.
    Generate sampling points and perform LSGL.
'''

import re
import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import LSGL.LoFTR as LoFTR
import time
import StreetView.Block as Block
import StreetView.spider as spider
from StreetView.spider import chekdir
from shapely.ops import nearest_points

def LSGL(mid_path,smd_pic,street_pic):
    '''Perform LSGL calculations and output results.
    Args:
        mid_path(str): Output folder for calculation results.
        smd_pic(str): Social media image folders that need to be calculated.
        street_pic(str): Folders of Street View images that need to be computed.
    '''
    result_path = mid_path + r"/res/"
    
    # Calling LoFTR match calculation
    LoFTR.process(smd_pic, street_pic, result_path)
    LoFTR.rating(result_path)
    LoFTR.write_res(result_path,0)
    x, y, res_score = LoFTR.bestpid(0, result_path, street_pic, 0)
    best_point = (x,y)
    time.sleep(0.5)
    print(" Matching results ".center(150 // 2,"*"))
    print("Best score is:",res_score)
    print("Coordinates are",best_point)
    return best_point

def search_street(all_generated_points,street_output):
    '''Retrieve the street view based on the input points.
    Args:
        all_generated_points(list): Sampled Point Sets.
        street_output(str): Output folder for Street View images.
    '''
    # Clean up the output folder
    chekdir(street_output)
    
    # Acquiring Street View Images with Sampled Point Sets
    spider.getPic(street_output, all_generated_points, "list")
    street_num = Block.chekres(street_output)
    
    # Calculate the acquired street view image points
    print("Total number of street view image pairs acquired: {}".format(street_num))
    
    # Check to see if Street View images have been acquired
    if street_num == 0:
        print('No street view image was obtained, please check the input coordinates!')

def search_road(osm_data,point,standardized_address,type,num):
    '''Sampling point generation using road names.
    Args:
        osm_data(geometry): Open Street Map data for the study area.
        point(Point): Coordinates of the sampling center point.
        standardized_address(str): Standardized sampling point addresses.
        type(int): Sampling type.
        num(int): sampling interval.
    Returns:
        all_generated_points(list): List with coordinates of sampling points.
    '''
    # Extracting road names from standardized addresses
    road_name_match = re.search(r"(?:市|县|区)([\w\s\d]+)", standardized_address)
    if road_name_match:
        road_name = road_name_match.group(1)
    else:
        road_name = standardized_address

    # Escape special characters in road_name
    road_name_escaped = re.escape(road_name)

    # Determine if the normalized address exists in the OSM.shp table of road names
    matched_road = osm_data[osm_data['name'].str.contains(road_name_escaped, na=False)]

    if not matched_road.empty:
        road_geometry = matched_road.iloc[0]['geometry']
    else:
        # Finding the nearest road
        nearest_geom = nearest_points(point, osm_data.unary_union)[1]
        nearest_road = osm_data[osm_data['geometry'].apply(lambda x: x.distance(nearest_geom) < 1e-8)].iloc[0]
        road_geometry = nearest_road['geometry']

    # If type is 1, isometric segmentation is done.
    if type == 0:
        all_generated_points = Block.cut_road_0(road_geometry,num)
    elif type == 1:
        all_generated_points = Block.cut_road_1(road_geometry,num)
    return all_generated_points

def search_building(osm_data, point, interval, radius):
    '''Sampling centered on the sampling point, based on the sampling radius and sampling spacing.
    Args:
        osm_data(geometry): Open Street Map data for the study area.
        point(list): Coordinates of the sampling center point.
        interval(int): Sampling interval.
        radius(int): Sampling radius.
    Returns:
        points(list): List with coordinates of sampling points.
    '''
  
    # Setting the coordinates of the sampling point
    lon, lat = point[0],point[1]

    # Creating buffer elements
    point = Point(lon, lat)
    buffer = point.buffer(radius * 0.00001) # conversion unit


    # Extracting roads intersecting a buffer using spatial operations
    buffer_geo = gpd.GeoDataFrame(geometry=[buffer], crs=4326)
    intersected_roads = gpd.overlay(osm_data, buffer_geo, how='intersection')

    # Sampling at set intervals
    spacing = interval * 0.00001
    points_on_lines = []
    for index, row in intersected_roads.iterrows():
        line = row.geometry
        distance = line.length
        num_points = int(distance/spacing)
        for i in range(num_points):
            point_on_line = line.interpolate(i*spacing)
            points_on_lines.append(point_on_line)
            
    points = []
    for point in points_on_lines:
      single_point = str(point).split("(")[-1].split(")")[0].split(" ")
      points.append((float(single_point[0]), float(single_point[1])))

    return points






