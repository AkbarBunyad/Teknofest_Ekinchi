from typing import List, Tuple, Dict

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from shapely import Point, Polygon
import rasterio
import geopandas as gpd
import torch

from models import dubois, SoilMoisturePredictor
from utils import (
    ANGLE,
    get_weather_forecast, 
    predict_soil_moisture, 
    get_field_sar,
    get_field_spec,
    get_water_required,
    update_map,
    format_spec,
    load_geojson,
    get_hectares)
from colors import transform2rgb, convert2image
from preprocessing import free_outliers


MODEL_PATH = './saved_models/model_cpu.pt'
PATH_IMAGE_DESCRIPTION = './data/sentinel_data/fields'

def load_model(model_path: str = MODEL_PATH):
    """Load Soil Moisture Predictor"""

    if model_path is None:
        model_path = MODEL_PATH

    checkpoints = torch.load(model_path)
    model = SoilMoisturePredictor()
    model.load_state_dict(checkpoints)

    return model

class Processor():
    def __init__(self, field_id: int = 1, device: str = 'cpu'):
        self.field_id = field_id
        self.model = load_model().to(device)
        self.path_image_description = PATH_IMAGE_DESCRIPTION

    def update_field_id(self, field_id: int):
        self.field_id = field_id

    def get_spec(self) -> np.ndarray:
        """The function to generate spectrum with soil moisture map and save"""
        spec, src = get_field_spec(field_id=self.field_id)
        sm_map, sm, src = self.get_sm_map()
        poly = load_geojson(field_id=self.field_id)

        spec_formatted = format_spec(spec=spec, sm_map=sm_map, poly=poly, src=src)
        spec_image = convert2image(spec_formatted.transpose([1, 2, 0]))
        
        spec_image.save('./images/field_{}.png'.format(self.field_id))

        insights = self.get_insights(sm=sm, poly=poly, src=src)

        return spec_image, insights
    
    def __predict(self) -> np.ndarray:
        _, temperature, precipitation = self.get_weather_forecast()
        soil_moisture_change = predict_soil_moisture(model=self.model, temperature=temperature, precipitation=precipitation)
        return soil_moisture_change
        
    def see_future(self) -> Tuple[List[float], List[float]]:
        sm, src = self.get_sm()
        sm_change = self.__predict()

        indexes = self.__get_indexes(sample=sm, src=src)
        
        water_required_total, water_required_per = get_water_required(soil_moisture_map=sm, indexes=indexes)

        water_totals = [water_required_total]
        water_pers = [water_required_per]
        for i in range(sm_change.size):
            sm = update_map(soil_moisture_map=sm, soil_moisture_change=sm_change[i])

            water_required_total, water_required_per = get_water_required(soil_moisture_map=sm, indexes=indexes)
            
            self.get_status_report(sm=sm)

            water_totals.append(water_required_total)
            water_pers.append(water_required_per)
        
        return water_totals, water_pers
    
    def get_insights(self, sm: np.ndarray, poly: Polygon, src: rasterio.io.DatasetReader):
        filepath = os.path.join(
            self.path_image_description, str(self.field_id), 'image_description', 'image_description.json'
            )
        
        data = json.load(open(filepath))
        
        indexes = self.__get_indexes(sample=sm, src=src)
        _, water_per = get_water_required(soil_moisture_map=sm, indexes=indexes)

        field_size = get_hectares(soil_moisture_map=sm, poly=poly, src=src)
        status = self.get_status_report(sm=sm)

        data['Hectare'].append('{} hectares'.format(field_size))
        data['Status'].append(status)

        if status == 'Bad':
            data['Problems'].append('Lack of water distribution')
            data['Recommendations'].append('{} kg/m^2 water is required'.format(round(water_per, 3)))

        else:
            data['Problems'].append('Water distribution is normal')
            data['Recommendations'].append('No irrigation is needed')
            
        return data

    def get_weather_forecast(self, days: int = 5) -> Tuple[List[str], List[float], List[float]]:
        return get_weather_forecast(field_id=self.field_id, days=days)

    def get_sm(self) -> np.ndarray:
        (vv, vh), src = get_field_sar(field_id=self.field_id)
        sm = dubois(vv=vv, vh=vh, angle=ANGLE)
        sm = free_outliers(sm[None], whis=1.5)[0]
        sm[sm < 0.2] = 0.2

        return sm, src
    
    def get_sm_map(self) -> np.ndarray:
        sm, src = self.get_sm()
        sm_map = transform2rgb(sm)
        return sm_map, sm, src
    
    def get_status_report(self, sm: np.ndarray) -> str:
        status = sm.mean()
      
        if status > 0.6:
            return 'Bad'

        elif status > 0.4:
            return 'Good'
        
        elif status > .3:
            return 'Normal'

        else:
            return 'Bad'
        
    def __get_indexes(self, sample: np.ndarray, src: rasterio.io.DatasetReader) -> np.ndarray:
        height, width = sample.shape
        poly = load_geojson(field_id=self.field_id)

        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    
        lons = np.array(xs)
        lats = np.array(ys)

        lons = lons.reshape(-1)
        lats = lats.reshape(-1)

        points = gpd.points_from_xy(lons, lats)
        indexes = np.where(points.within(poly))[0]
        return indexes

if __name__ == '__main__':
    
    # Create a processor for all the fields
    processor = Processor(device='cpu')

    # Provide field_id between 1 and 4
    processor.update_field_id(field_id=4)

    # The function saves an image that can be used in website.
    _, insights = processor.get_spec()
    
    # insights variable contains the information about Hectare, Problems, Status, and Recommendations
    print(insights.keys())
    print(insights)

    # The function returns 5-day weather forecasting by default. 
    # Datetime, temperature, and precipitation are returned
    date_time, temperature, precipitation = processor.get_weather_forecast(days=5)
    
    print('date_time:', date_time)
    print('temperature:', temperature)
    print('precipitation:', precipitation)

    # Those are the predicted required amount of water over the next 5-days.
    # water_totals --> total amount of water required for the field in kg
    # water_pers --> amount of water per m^2 for the field in kg/m^2
    water_totals, water_pers = processor.see_future()
    print('water total:', water_totals)
    print('water per:', water_pers)

    print()
    print(insights['Problems'])
    print(insights['Hectare'])