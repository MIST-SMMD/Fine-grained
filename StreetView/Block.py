''' Methods of processing OSM data.
'''

import csv
import os
import pyproj
import shutil
import matplotlib.pyplot as plt
from shapely.ops import transform
from matplotlib.patches import Rectangle
from shapely.geometry import Point, LineString, MultiLineString

def chekdir(path):
    '''Check if the folder has been created.
    Args:
        path(str): Raw processing path for input.
    '''
    try:
        os.mkdir(path)
    except:
        return 0

def chekres(path):
    '''Check the number of street view images acquired.
    Args:
        path(str): Folders where Street View images are stored.
    Returns:
        Number of street view images acquired.
    '''
    path = path + '/downloadPic/'
    if  len(os.listdir(path)) > 0:
        return len(os.listdir(path))
    else:
        return 0

def cut_road_0(road_geometry,distance):
    '''Sampling by setting the sampling interval.
    Args:
        road_geometry(geometry): OSM data for sampling.
        distance(int): sampling interval.
    Returns:
        all_generated_points(list): List with coordinates of sampling points.
    '''
    all_generated_points = []

    # Defining Conversion Functions
    world_mercator = pyproj.CRS('EPSG:3395')
    wgs84 = pyproj.CRS('EPSG:4326')
    WM_2_wgs84 = pyproj.Transformer.from_crs(world_mercator, wgs84, always_xy=True).transform
    wgs84_2_WM = pyproj.Transformer.from_crs(wgs84, world_mercator, always_xy=True).transform
    def cut_line(line, step):
        points = []

        # Generate points every step meter from the start of the road.
        dist_along_line = step
        while dist_along_line < line.length:
            point = line.interpolate(dist_along_line)
            wgs84_point = (point.x, point.y)
            points.append(wgs84_point)
            dist_along_line += line.project(point) - dist_along_line + step
        return points
    
    # Determining if a road geometry is a multipart geometry
    if isinstance(road_geometry, MultiLineString):
        i = 0
        for line in road_geometry.geoms:
            
            # Take the starting point
            if i == 0:
                start_point = line.coords[0]
                all_generated_points.append((start_point[0], start_point[1]))
            
            # Projection to the projected coordinate system WGS 1984 World Mercator to use linear units
            line = transform(wgs84_2_WM, line)
            
            # Processing each line segment
            points = cut_line(line, distance)
            
            # Convert coordinates to target coordinate system
            points_wgs84 = [WM_2_wgs84(x, y) for x, y in points]
            all_generated_points.extend(points_wgs84)
            end_point = line.coords[-1]
            all_generated_points.append((end_point[0], end_point[1]))
            i += 1
    else:
        # Beginning and end of the road to access
        start_point, end_point = road_geometry.coords[0], road_geometry.coords[-1]
        
        # Add road start and end points
        all_generated_points.append((start_point[0], start_point[1]))
        all_generated_points.append((end_point[0], end_point[1]))
        points = cut_line(road_geometry, distance)
        points_wgs84 = [WM_2_wgs84(x, y) for x, y in points]
        all_generated_points.extend(points_wgs84)

    return all_generated_points

def cut_road_1(road_geometry,num_points):
    '''Sampling by setting the number of sampling points.
    Args:
        road_geometry(geometry): OSM data for sampling.
        num_points(int): Number of sampling points.
    Returns:
        all_generated_points(list): List with coordinates of sampling points.
    '''
    all_generated_points = []

    # Generate points by customizing the split into equal parts along the roadway
    road_length = road_geometry.length
    split_points = [road_geometry.interpolate(i * road_length / num_points) for i in range(1, num_points)]

    # Returns the WGS1984 coordinates of these points
    wgs84_points = [(point.y, point.x) for point in split_points]
    all_generated_points.extend(wgs84_points)

    return all_generated_points
