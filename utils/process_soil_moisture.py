import sys
sys.path.append('../')

import numpy as np
import os
from models import dubois
from preprocessing import free_outliers, min_max_scale, calibrate
import rasterio
import geojson
from shapely import Polygon, Point
from skimage.exposure import equalize_hist

ANGLE = 65 * np.pi / 180
K = 1

ROOT = './data/sentinel_data/fields/'

def get_field_sar(field_id: int, root: str = ROOT) -> np.ndarray:
    """Returns Synthetic Aparture Radar (SAR) view of the field with preprocessing"""
    field_id = str(field_id)
    field_path = os.path.join(root, field_id)
    field_path_sar = os.path.join(field_path, 'sar', 'vv_vh.tiff')
    
    with rasterio.open(field_path_sar) as src:
        field_sar = src.read()

    field_sar = free_outliers(field_sar, whis=1.5)
    field_sar = min_max_scale(field_sar, bit_size=16)
    field_sar_calibrated = calibrate(field_sar, angle=ANGLE, K=K)
    
    return field_sar_calibrated, src

def get_field_spec(field_id: int, root: str = ROOT) -> np.ndarray:
    """Returns multispectrum view of the field"""
    field_id = str(field_id)
    field_path = os.path.join(root, str(field_id))
    field_path_spec = os.path.join(field_path, 'spec', 'true_color.tiff')
    
    with rasterio.open(field_path_spec) as src:
        field_spec = src.read() / 2 ** 16 
    
    field_spec = equalize_hist(field_spec)
    field_spec = (field_spec * 255).astype(np.uint16)

    return field_spec, src

def load_geojson(field_id: int, root: str = ROOT) -> Polygon:
    """Loading polygon for the customer's field"""
    field_path = os.path.join(root, str(field_id))
    field_path_coord = os.path.join(field_path, 'coords', 'field.geojson')

    data = geojson.load(open(field_path_coord))
    coords = data['coordinates']
    poly = Polygon(coords[0])

    return poly

def format_spec(spec: np.ndarray, sm_map: np.ndarray, poly: Polygon, src: rasterio.io.DatasetReader) -> np.ndarray:
    spec = spec.copy()
    
    height, width = spec.shape[1:]
    
    shape_spec = spec.shape
    shape_sm = sm_map.shape

    spec = spec.reshape(3, -1)
    sm_map = sm_map.reshape(3, -1)

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    
    lons = np.array(xs)
    lats = np.array(ys)

    lons = lons.reshape(-1)
    lats = lats.reshape(-1)

    for i in range(lons.size):
        point = Point([lons[i], lats[i]])
        if point.within(poly):
            spec[:, i] = sm_map[:, i]
    
    spec = spec.reshape(3, *shape_spec[1:])
    sm_map = sm_map.reshape(3, *shape_sm[1:])

    return spec

def get_water_required(soil_moisture_map: np.ndarray, indexes: np.ndarray, soil_density: float = 1040) -> float:
    """Water requirement estimation"""
    soil_volume = 10 * 10 * 0.1 # m^3
    soil_mass = soil_density * soil_volume

    soil_moisture_map_defiency = 0.3 - soil_moisture_map
    soil_moisture_map_defiency[soil_moisture_map_defiency < 0] = 0

    water_defiency = soil_moisture_map_defiency * soil_mass
    
    water_required_total = water_defiency.reshape(-1)[indexes].sum()
    water_required_per = water_required_total / indexes.size / 100 # kg / m^2
    
    return float(water_required_total), float(water_required_per)

def update_map(soil_moisture_map: np.ndarray, soil_moisture_change: float, threshold_min: float = 0.2, threshold_max: float = 0.8) -> np.ndarray:
    """Estimate potential changes based on the weather forecasting"""
    soil_moisture_map = soil_moisture_map + soil_moisture_change
    soil_moisture_map[soil_moisture_map < threshold_min] = threshold_min
    soil_moisture_map[soil_moisture_map > threshold_max] = threshold_max
    return soil_moisture_map

def get_hectares(soil_moisture_map: np.ndarray, poly: Polygon, src: rasterio.io.DatasetReader):

    if soil_moisture_map.ndim == 3:
        soil_moisture_map = soil_moisture_map[0]

    height, width = soil_moisture_map.shape

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    
    lons = np.array(xs)
    lats = np.array(ys)

    lons = lons.reshape(-1)
    lats = lats.reshape(-1)

    index_count = 0
    for i in range(lons.size):
        point = Point([lons[i], lats[i]])
        if point.within(poly):
            index_count += 1
    
    return index_count * 100 / 10 ** 4