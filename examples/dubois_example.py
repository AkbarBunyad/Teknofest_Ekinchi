import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..__init__ import preprocessing
# preprocessing import free_outliers, min_max_scale, calibrate
# from ..models import dubois


if __name__ == '__main__':
    root = '../sentinel_data/fields/field_1'
    spec_path = os.path.join(root, 'spec')
    sar_path = os.path.join(root, 'sar')
    
    angle = 65 / 180 * np.pi
    K = 1
    wave_length = 5.625

    true_color_path = os.path.join(spec_path, 'true_color.tiff')

    filenames = os.listdir(sar_path)
    
    if '_VV_' in filenames[0]:
        vv_name = filenames[0]
        vh_name = filenames[1]

    else:
        vv_name = filenames[1]
        vh_name = filenames[0]

    vv_path = os.path.join(sar_path, vv_name)
    vh_path = os.path.join(sar_path, vh_name)
    
    with rasterio.open(vv_path) as src:
        vv = src.read()

    with rasterio.open(vh_path) as src:
        vh = src.read()
    
    vv_vh = np.concatenate((vv, vh), axis=0)
    vv_vh = free_outliers(vv_vh, whis=1.5)
    vv_vh = min_max_scale(vv_vh, bit_size=16)
    vv_vh = calibrate(vv_vh, angle=angle, K=K)
    
    vv, vh = vv_vh

    # Retrieving soil moisture back from radar imagery
    sm = dubois(vv=vv, vh=vh, angle=angle, wave_length=wave_length)

    # Removing negatives
    sm[sm < 0] = sm[sm > 0].min()
    print(sm.shape)

    sns.displot(sm.reshape(-1))
    plt.show()

    plt.imshow(sm)
    plt.show()

    # with rasterio.open(true_color_path) as src:
    #     true_color = src.read() / 2 ** 16 * 255
    #     true_color = true_color.astype(np.uint16)

    #     print(true_color.shape)
    #     true_color[0], true_color[2] = true_color[2], true_color[0]
    #     true_color = true_color.transpose([1, 2, 0])

    # plt.imshow(true_color)
    # plt.show()
