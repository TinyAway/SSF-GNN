import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal


indice_names = ["NDVI", "FAI", "NDWI", "MNDWI", "MGTI", "NDOI", "NDBI", "BAI", "NBR", "EVI", "SAVI", "TWI", "CMI",
                "ARVI", "NDRE"]

def calculate_ndvi(red_band, nir_band):
    """NDVI"""
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-7)  # 1e-7 避免除以零
    return ndvi

def calclate_fai(nir_band, red_band, swir_band1):
    """FAI"""
    fai = nir_band - (red_band + (swir_band1 - red_band) * (177 / 945))
    return fai

def calculate_ndwi(green_band, nir_band):
    """NDWI"""
    ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-7)
    return ndwi

def calculate_mndwi(green_band, swir_band):
    """MNDWI"""
    mndwi = (green_band - swir_band) / (green_band + swir_band + 1e-7)
    return mndwi

def calculate_mgti(green_band, blue_band, red_band):
    """MGTI"""
    mgti = green_band - blue_band - (red_band - blue_band) * (70 / 175)
    return mgti

def calculate_ndoi(red_band, b8a_band, swir_band):
    """NDOI"""
    ndoi = (red_band - b8a_band) / (red_band + b8a_band + 1e-7) + swir_band
    return ndoi

def calculate_ndbi(swir_band, nir_band):
    """NDBI"""
    ndbi = (swir_band - nir_band) / (swir_band + nir_band + 1e-7)
    return ndbi

def calculate_bai(red_band, nir_band):
    """BAI"""
    bai = 1 / (np.square(0.1 - red_band) + np.square(0.06 - nir_band) + 1e-7)
    return bai

def calculate_nbr(b8a_band, swir2_band):
    """NBR"""
    nbr = (b8a_band - swir2_band) / (b8a_band + swir2_band + 1e-7)
    return nbr

def calculate_evi(blue_band, red_band, nir_band):
    """EVI"""
    evi = 2.5 * (nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * blue_band + 1 + 1e-7)
    return evi

def calculate_savi(red_band, nir_band, L=0.5):
    """SAVI"""
    savi = (nir_band - red_band) * (1 + L) / (nir_band + red_band + L + 1e-7)
    return savi

def calculate_twi(red_band, swir_band):
    """TWI"""
    twi = red_band - swir_band
    return twi

def calculate_cmi(green_band, blue_band, swir1_band):
    """CMI"""
    cmi = green_band - blue_band - (swir1_band - blue_band) * (70 / 1120)
    return cmi

def calculate_arvi(nir_band, red_band, blue_band):
    """ARVI"""
    arvi = (nir_band - 2 * red_band + blue_band) / (nir_band + 2 * red_band + blue_band + 1e-7)
    return arvi

def calculate_ndre(nir_band, rededge_band):
    """NDRE"""
    ndre = (nir_band - rededge_band) / (nir_band + rededge_band + 1e-7)
    return ndre

def calculate_rwi(green_band, rededge1_band, nir_band, b8a_band, swir2_band):
    rwi = ((green_band + rededge1_band) - (nir_band + b8a_band + swir2_band)) / (green_band + rededge1_band + nir_band + b8a_band + swir2_band + 1e-7)
    return rwi

def compute_indices(img):
    img = np.clip(img, 0, 1)
    indices = []
    blue_band = img[1]
    green_band = img[2]
    red_band = img[3]
    rededge1_band = img[4]
    rededge2_band = img[5]
    rededge3_band = img[6]
    nir_band = img[7]
    b8a_band = img[8]
    swir_band1 = img[10]
    swir_band2 = img[11]

    ndvi = calculate_ndvi(red_band, nir_band)
    indices.append(ndvi)
    fai = calclate_fai(nir_band, red_band, swir_band1)
    indices.append(fai)
    ndwi = calculate_ndwi(green_band, nir_band)
    indices.append(ndwi)
    mndwi1 = calculate_mndwi(green_band, swir_band1)
    indices.append(mndwi1)
    mndwi2 = calculate_mndwi(green_band, swir_band2)
    indices.append(mndwi2)
    mgti = calculate_mgti(green_band, blue_band, red_band)
    indices.append(mgti)
    ndoi1 = calculate_ndoi(red_band, b8a_band, swir_band1)
    indices.append(ndoi1)
    ndoi2 = calculate_ndoi(red_band, b8a_band, swir_band2)
    indices.append(ndoi2)
    ndbi1 = calculate_ndbi(swir_band1, nir_band)
    indices.append(ndbi1)
    ndbi2 = calculate_ndbi(swir_band2, nir_band)
    indices.append(ndbi2)
    nbr = calculate_nbr(b8a_band, swir_band2)
    indices.append(nbr)
    savi = calculate_savi(red_band, nir_band)
    indices.append(savi)
    twi1 = calculate_twi(red_band, swir_band1)
    indices.append(twi1)
    twi2 = calculate_twi(red_band, swir_band2)
    indices.append(twi2)
    cmi = calculate_cmi(green_band, blue_band, swir_band1)
    indices.append(cmi)
    arvi = calculate_arvi(nir_band, red_band, blue_band)
    indices.append(arvi)
    ndre1 = calculate_ndre(nir_band, rededge1_band)
    indices.append(ndre1)
    ndre2 = calculate_ndre(nir_band, rededge2_band)
    indices.append(ndre2)
    ndre3 = calculate_ndre(nir_band, rededge3_band)
    indices.append(ndre3)
    rwi = calculate_rwi(green_band, rededge1_band, nir_band, b8a_band, swir_band2)
    indices.append(rwi)
    indices = np.stack(indices)
    return indices


if __name__ == "__main__":
    import cv2
    img = gdal.Open('19598.tif')
    img = img.ReadAsArray().astype(np.float32)  # 获取数据
    for i, ii in enumerate(img):
        rgb = cv2.cvtColor(ii, cv2.COLOR_GRAY2RGB)
        plt.imsave("band{}.png".format(str(i)), rgb)



