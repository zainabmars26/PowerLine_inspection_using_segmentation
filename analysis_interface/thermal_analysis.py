import ctypes
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import exifread

DLL_PATH = r"D:\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\libdirp.dll"

# Global cache for raw matrices to allow fast UI click lookups: { rjpeg_path: temp_matrix }
TEMP_MATRIX_CACHE = {}

try:
    dirp_dll = ctypes.CDLL(DLL_PATH)
    dirp_dll.dirp_create_from_rjpeg.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
    dirp_dll.dirp_create_from_rjpeg.restype = ctypes.c_int
    dirp_dll.dirp_measure_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    dirp_dll.dirp_measure_ex.restype = ctypes.c_int
    dirp_dll.dirp_destroy.argtypes = [ctypes.c_void_p]
    dirp_dll.dirp_destroy.restype = ctypes.c_int
except Exception as e:
    dirp_dll = None
    print(f"Warning: Failed to load DIRP DLL from {DLL_PATH}: {e}")

class_colors = {
    'background': (0, 0, 0),
    'insulator_down': (0, 255, 0),
    'insulator_up': (255, 0, 0),
    'tower_structure': (0, 255, 255),
    'wires': (255, 0, 255),
    'clamps': (0, 255, 255),
    'fixer': (255, 128, 0),
    'insulator_glass': (128, 128, 128)
}

# =========================================================================
# HELPER: Save exact pixel-for-pixel thermal rendering
# =========================================================================
def save_analyzed_thermal_image(thermal_matrix, output_path, raw_width, raw_height, anomalies=None):
    """Saves thermal result at the exact 1:1 resolution (1280x1024) with anomaly boxes."""
    # Normalize thermal matrix to 0-255 uint8 range
    valid_mask = ~np.isnan(thermal_matrix)
    norm_matrix = np.zeros((raw_height, raw_width), dtype=np.uint8)

    if np.any(valid_mask):
        min_val, max_val = np.nanmin(thermal_matrix), np.nanmax(thermal_matrix)
        if max_val > min_val:
            scaled = (thermal_matrix - min_val) / (max_val - min_val) * 255.0
            norm_matrix = np.nan_to_num(scaled, nan=0).astype(np.uint8)

    # Apply Inferno colormap
    colored_img = cv2.applyColorMap(norm_matrix, cv2.COLORMAP_INFERNO)

    # Dark background for unsegmented regions
    colored_img[~valid_mask] = [30, 30, 30]

    # Draw anomaly bounding boxes and text directly onto image
    if anomalies:
        for anomaly in anomalies:
            x, y, w, h = anomaly['box']
            diff = anomaly['diff']

            # Draw cyan bounding box (BGR: 255, 255, 0)
            cv2.rectangle(colored_img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Draw label box above the rectangle
            label = f"|dT|={diff:.1f}C"
            label_y = max(20, y - 8)
            
            # Label background box
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(colored_img, (x, label_y - text_h - 4), (x + text_w + 4, label_y + 2), (0, 0, 0), -1)
            
            # Label text
            cv2.putText(colored_img, label, (x + 2, label_y - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # Save exact 1:1 pixel output
    cv2.imwrite(output_path, colored_img)

def get_raw_temperature_matrix(rjpeg_path):
    if rjpeg_path in TEMP_MATRIX_CACHE:
        return TEMP_MATRIX_CACHE[rjpeg_path]

    if dirp_dll is None:
        raise RuntimeError("DIRP SDK DLL is not loaded.")
    
    with open(rjpeg_path, 'rb') as f:
        rjpeg_bytes = f.read()
    
    buffer_len = len(rjpeg_bytes)
    rjpeg_buffer = (ctypes.c_uint8 * buffer_len).from_buffer_copy(rjpeg_bytes)
    
    img = cv2.imread(rjpeg_path)
    height, width = img.shape[:2]
    
    handle = ctypes.c_void_p()
    ret = dirp_dll.dirp_create_from_rjpeg(rjpeg_buffer, buffer_len, ctypes.byref(handle))
    if ret != 0:
        raise RuntimeError(f"SDK Error: {ret}")

    temp_buffer = (ctypes.c_float * (width * height))()
    ret_measure = dirp_dll.dirp_measure_ex(handle, temp_buffer, ctypes.sizeof(temp_buffer))
    dirp_dll.dirp_destroy(handle)

    if ret_measure != 0:
        raise RuntimeError(f"Measurement Error: {ret_measure}")
    
    matrix = np.ctypeslib.as_array(temp_buffer).reshape((height, width))
    TEMP_MATRIX_CACHE[rjpeg_path] = matrix
    return matrix

def get_temperature_at_pixel(rjpeg_path, x, y):
    """Returns the exact float temperature (°C) at specific (x, y) image coordinates."""
    try:
        matrix = get_raw_temperature_matrix(rjpeg_path)
        h, w = matrix.shape
        if 0 <= x < w and 0 <= y < h:
            return float(matrix[int(y), int(x)])
    except Exception as e:
        print(f"Failed to fetch pixel temperature: {e}")
    return None

def get_thermal_center_gps(rjpeg_path):
    try:
        with open(rjpeg_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat_vals = tags['GPS GPSLatitude'].values
            lat_ref = str(tags.get('GPS GPSLatitudeRef', 'N'))
            lon_vals = tags['GPS GPSLongitude'].values
            lon_ref = str(tags.get('GPS GPSLongitudeRef', 'E'))

            lat = float(lat_vals[0]) + float(lat_vals[1])/60.0 + (float(lat_vals[2].num)/float(lat_vals[2].den))/3600.0
            if lat_ref == 'S': lat = -lat

            lon = float(lon_vals[0]) + float(lon_vals[1])/60.0 + (float(lon_vals[2].num)/float(lon_vals[2].den))/3600.0
            if lon_ref == 'W': lon = -lon

            alt = 0.0
            if 'GPS GPSAltitude' in tags:
                alt_val = tags['GPS GPSAltitude'].values[0]
                alt = float(alt_val.num) / float(alt_val.den)

            return f"Center GPS: {lat:.6f}°, {lon:.6f}° | Alt: {alt:.1f}m"

        with open(rjpeg_path, 'rb') as f:
            content = f.read()
            xmp_start = content.find(b'<x:xmpmeta')
            xmp_end = content.find(b'</x:xmpmeta>')

            if xmp_start != -1 and xmp_end != -1:
                xmp_data = content[xmp_start:xmp_end+12].decode('utf-8', errors='ignore')
                lat, lon, alt = None, None, None
                for line in xmp_data.split('\n'):
                    if 'GpsLatitude=' in line or 'drone-dji:GpsLatitude=' in line:
                        lat = float(line.split('=')[1].strip('" />\r'))
                    if 'GpsLongitude=' in line or 'drone-dji:GpsLongitude=' in line:
                        lon = float(line.split('=')[1].strip('" />\r'))
                    if 'AbsoluteAltitude=' in line or 'drone-dji:AbsoluteAltitude=' in line:
                        alt = float(line.split('=')[1].strip('" />\r'))

                if lat is not None and lon is not None:
                    alt_str = f"{alt:.1f}m" if alt is not None else "N/A"
                    return f"Center GPS: {lat:.6f}°, {lon:.6f}° | Alt: {alt_str}"

        return "Center GPS: Not Found"
    except Exception as e:
        return f"Center GPS: Error ({str(e)})"

def detect_and_plot_anomalies(rjpeg_file, mask_file, output_dir, threshold_deg=15.0, min_area=3):
    temp_matrix = get_raw_temperature_matrix(rjpeg_file)
    gps_str = get_thermal_center_gps(rjpeg_file)
    img_h, img_w = temp_matrix.shape

    mask_bgr = cv2.imread(mask_file)
    if mask_bgr is None:
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    if mask_bgr.shape[:2] != temp_matrix.shape:
        mask_bgr = cv2.resize(mask_bgr, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    powerline_bool_mask = np.any(mask_rgb > 0, axis=-1)
    isolated_temp_matrix = np.where(powerline_bool_mask, temp_matrix, np.nan)
    
    anomalies = []
    for class_name, target_rgb in class_colors.items():
        if class_name == 'background':
            continue
            
        component_mask = np.all(mask_rgb == target_rgb, axis=-1)
        component_temps = temp_matrix[component_mask]
        
        if len(component_temps) == 0:
            continue
            
        class_mean = np.mean(component_temps)
        anomaly_mask = component_mask & (np.abs(temp_matrix - class_mean) > threshold_deg)
        anomaly_mask_uint8 = anomaly_mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(anomaly_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pad = 15

        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
                
                region_temps = temp_matrix[y:y+h, x:x+w]
                peak_temp = np.max(region_temps) if len(region_temps) > 0 else class_mean
                temp_diff = np.max(np.abs(region_temps - class_mean))
                
                anomalies.append({
                    'class': class_name,
                    'box': (x1, y1, x2 - x1, y2 - y1),
                    'peak_temp': peak_temp,
                    'diff': temp_diff
                })

# Save output image using exact sensor pixel dimensions (1280x1024)
    base_name = os.path.splitext(os.path.basename(rjpeg_file))[0]
    out_thermal_path = os.path.join(output_dir, f"{base_name}_thermal_res.png")
    
    save_analyzed_thermal_image(
        thermal_matrix=isolated_temp_matrix, 
        output_path=out_thermal_path, 
        raw_width=img_w, 
        raw_height=img_h,
        anomalies=anomalies
    )

    return anomalies, out_thermal_path

def direct_project_thermal_to_wide(thermal_img_path, wide_img_path, anomaly_boxes, output_path):
    thermal_img = cv2.imread(thermal_img_path)
    wide_img = cv2.imread(wide_img_path)

    if wide_img is None:
        if thermal_img is not None:
            wide_img = np.zeros_like(thermal_img)
        else:
            wide_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    wide_result = wide_img.copy()

    if not anomaly_boxes:
        cv2.putText(
            wide_result, "STATUS: NO ANOMALY DETECTED", 
            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
        )
        cv2.imwrite(output_path, wide_result)
        return output_path

    th_h, th_w = thermal_img.shape[:2]
    w_h, w_w = wide_img.shape[:2]

    scale_factor = 0.4615
    mapped_w = w_w * scale_factor
    mapped_h = mapped_w * (th_h / th_w)

    offset_x = (w_w - mapped_w) / 2.0
    offset_y = (w_h - mapped_h) / 2.0

    for box_info in anomaly_boxes:
        x, y, w, h = box_info['box']

        x_wide = int(offset_x + (x / th_w) * mapped_w)
        y_wide = int(offset_y + (y / th_h) * mapped_h)
        w_wide = int((w / th_w) * mapped_w)
        h_wide = int((h / th_h) * mapped_h)

        pad_x, pad_y = 60, 40
        x_wide_padded = max(0, x_wide - (pad_x // 2))
        y_wide_padded = max(0, y_wide - (pad_y // 2))

        cv2.rectangle(
            wide_result, 
            (x_wide_padded, y_wide_padded), 
            (x_wide_padded + w_wide + pad_x, y_wide_padded + h_wide + pad_y), 
            (255, 255, 0), 3
        )

    cv2.imwrite(output_path, wide_result)
    return output_path

def process_single_pair(rjpeg_file, mask_file, wide_file, output_dir, threshold_deg=18.0):
    os.makedirs(output_dir, exist_ok=True)
    anomalies, out_thermal_path = detect_and_plot_anomalies(
        rjpeg_file, mask_file, output_dir, threshold_deg=threshold_deg
    )
    
    base_name = os.path.splitext(os.path.basename(rjpeg_file))[0]
    out_rgb_path = os.path.join(output_dir, f"{base_name}_rgb_res.png")
    
    direct_project_thermal_to_wide(rjpeg_file, wide_file, anomalies, out_rgb_path)
    return out_thermal_path, out_rgb_path