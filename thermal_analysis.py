import os
import ctypes
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from collections import defaultdict
from depth_anything_v2.dpt import DepthAnythingV2
import torch
import matplotlib
import exifread
from scipy.ndimage import distance_transform_edt


DLL_PATH = r"D:\dji_thermal_sdk_v1.8_20250829\utility\bin\windows\release_x64\libdirp.dll"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1024
INPUT_THERMAL = r"C:\Users\Zainab.Alawneh\Desktop\thermal_results_edited 2\d5032f43-DJI_20260730095707_0023_T.JPG"
INPUT_MASK = r"C:\Users\Zainab.Alawneh\Desktop\thermal_results_edited 2\d5032f43-DJI_20260730095707_0023_T.png"
OUTPUT_CSV = r"D:\transmition-line inspection\Depth-Anything-V2\temperature_output.csv"


directory = os.path.dirname(INPUT_THERMAL)
image_name = os.path.splitext(os.path.basename(INPUT_THERMAL))[0]



DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
    
class_colors = {
    'background': (0, 0, 0),
    'insulator_down': (0, 255, 0),
    'insulator_up': (255, 0, 0),
    'tower_structure': (0, 255, 255),
    'wires': (255, 0, 255),
    'clamps': (255, 255, 0),
    'fixer': (255, 128, 0),
    "insulator_glass":(128,128,128)
}

marking_colors = {
    'insulator_down': (0, 255, 0),
    'insulator_up': (0, 0, 255),
    'tower_structure': (0, 255, 255),
    'wires': (255, 0, 255),
    'clamps': (255, 255, 0),
    'fixer': (128, 255, 0),
    "insulator_glass":(128,128,128)
}



def extract_dji_thermal_accurate(image_path, dll_path):
    """Extract actual radiometric temperature data using DJI SDK"""
    
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DJI SDK library not found at: {dll_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Thermal image not found at: {image_path}")

    print("Loading DJI DIRP library...")
    dji_lib = ctypes.CDLL(dll_path)

    # Define function signatures
    dji_lib.dirp_create_from_rjpeg.restype = ctypes.c_int
    dji_lib.dirp_create_from_rjpeg.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), 
        ctypes.c_int32, 
        ctypes.POINTER(ctypes.c_void_p)
    ]

    dji_lib.dirp_measure.restype = ctypes.c_int
    dji_lib.dirp_measure.argtypes = [
        ctypes.c_void_p, 
        ctypes.POINTER(ctypes.c_int16), 
        ctypes.c_int32
    ]

    dji_lib.dirp_destroy.restype = ctypes.c_int
    dji_lib.dirp_destroy.argtypes = [ctypes.c_void_p]

    # Read R-JPEG file
    print("Reading thermal image...")
    with open(image_path, 'rb') as f:
        rjpeg_bytes = f.read()
    
    file_size = len(rjpeg_bytes)
    data_buffer = (ctypes.c_uint8 * file_size).from_buffer(bytearray(rjpeg_bytes))

    # Create DIRP handle
    print("Parsing R-JPEG metadata...")
    handle = ctypes.c_void_p()
    ret_create = dji_lib.dirp_create_from_rjpeg(data_buffer, file_size, ctypes.byref(handle))
    
    if ret_create != 0:
        raise RuntimeError(f"Failed to create DIRP handle. Error code: {ret_create}")

    # Allocate buffer for temperature data
    pixel_count = IMAGE_WIDTH * IMAGE_HEIGHT
    temp_buffer_size = pixel_count * ctypes.sizeof(ctypes.c_int16)
    temp_buffer = (ctypes.c_int16 * pixel_count)()

    # Extract measurements
    print("Extracting temperature measurements...")
    ret_measure = dji_lib.dirp_measure(handle, temp_buffer, temp_buffer_size)
    if ret_measure != 0:
        dji_lib.dirp_destroy(handle)
        raise RuntimeError(f"Failed to extract measurements. Error code: {ret_measure}")

    # Cleanup
    dji_lib.dirp_destroy(handle)

    # Convert to Celsius matrix
    raw_array = np.frombuffer(temp_buffer, dtype=np.int16)
    celsius_matrix = raw_array.astype(np.float32) / 10.0
    celsius_matrix = celsius_matrix.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

    return celsius_matrix

# ==================== ORIENTATION & MOMENT ANALYSIS ====================

def get_major_axis_angle(mask):
    """
    Calculates the principal axis angle (in degrees, [-90, 90]) 
    using image moments (PCA).
    """
    moments = cv2.moments(mask.astype(np.uint8))
    if moments['m00'] == 0:
        return 0.0

    mu20 = moments['mu20'] / moments['m00']
    mu02 = moments['mu02'] / moments['m00']
    mu11 = moments['mu11'] / moments['m00']

    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    return np.degrees(angle_rad)

def get_tower_orientation(full_mask):
    """
    Finds yellow tower pixels and calculates main reference angle.
    """
    tower_color = class_colors['tower_structure']
    tower_mask = extract_class_mask(full_mask, tower_color)
    if np.sum(tower_mask) == 0:
        return 0.0
    return get_major_axis_angle(tower_mask)


def calculate_relative_orientation(component_mask, tower_angle):
    """
    Determines if insulator is 'vertical' or 'horizontal' relative to yellow tower.
    """
    insulator_angle = get_major_axis_angle(component_mask)
    
    diff = abs(insulator_angle - tower_angle) % 180
    if diff > 90:
        diff = 180 - diff

    return "vertical" if diff <= 45.0 else "horizontal"



def extract_class_mask(mask, class_color):
    """Extract binary mask for a specific class"""
    lower = np.array(class_color)
    upper = np.array(class_color)
    mask_binary = cv2.inRange(mask, lower, upper)
    return mask_binary

def label_connected_components(mask_binary):
    """Label connected components in mask"""
    labeled_array, num_features = ndimage.label(mask_binary)
    return labeled_array, num_features

def analyze_component(component_mask, temperatures, component_id, class_name):
    """Analyze a single component"""
    pixels = temperatures[component_mask > 0]
    
    if len(pixels) == 0:
        return None
    
    max_temp = np.max(pixels)
    max_idx = np.unravel_index(
        np.argmax(component_mask * temperatures),
        temperatures.shape
    )
    
    stats = {
        'class': class_name,
        'component_id': component_id,
        'max_temp': max_temp,
        'min_temp': np.min(pixels),
        'mean_temp': np.mean(pixels),
        'median_temp': np.median(pixels),
        'std_dev': np.std(pixels),
        'pixel_count': len(pixels),
        'max_location': (max_idx[1], max_idx[0])
    }
    
    return stats



def calculate_insulator_angle(component_mask):
    """
    Calculates the orientation angle of a detected insulator mask in degrees relative to vertical (0 degrees).
    """
    # Find contour of the component mask
    contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = contours[0]
    if len(cnt) < 5:  # Need at least 5 points for minAreaRect/fitEllipse
        return 0.0

    # Get minimum area rotated bounding box
    rect = cv2.minAreaRect(cnt)
    (center_x, center_y), (width, height), angle = rect


    if width < height:
        angle_from_vertical = angle
    else:
        angle_from_vertical = angle + 90.0 if angle < 0 else angle - 90.0

    
    return angle_from_vertical



def analyze_component_simple(component_mask, temperatures, component_id, class_name, tower_angle=0.0):
    """
    Erodes core to find hotspots and evaluates orientation relative to tower.
    """

    print(f"----------------id -----------------{class_name}")

    # Erode aggressively to remove ALL boundary contamination
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    core_mask = cv2.erode(component_mask.astype(np.uint8), kernel, iterations=1)

    # if np.sum(core_mask) < 200:
    if class_name in ['insulator_up', 'insulator_down', 'tower_structure']:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        core_mask = cv2.erode(component_mask.astype(np.uint8), kernel, iterations=2)
    
    if np.sum(core_mask) < 5:
        return None  # Can't get pure core
    
    pixels = temperatures[core_mask > 0]
    
    # Find max ONLY in the pure eroded core
    max_idx = np.unravel_index(np.argmax(pixels), core_mask.shape)
    angle = calculate_insulator_angle(component_mask)
    # But get coordinates in the original thermal image
    y, x = np.where(core_mask)
    max_in_core = np.argmax(temperatures[core_mask > 0])
    max_y = y[max_in_core]
    max_x = x[max_in_core]
    
    orientation = "N/A"
    if 'insulator' in class_name:
        orientation = calculate_relative_orientation(component_mask, tower_angle)
    
    return {
        'class': class_name,
        'component_id': component_id,
        'max_temp': np.max(pixels),  # Hottest in PURE CORE
        'mean_temp': np.mean(pixels),
        'median_temp': np.median(pixels),
        'min_temp': np.min(pixels),
        'std_dev': np.std(pixels),
        'pixel_count': len(pixels),
        'max_location': (max_x, max_y),
        'angle': angle,
        'orientation': orientation
    }


def process_class(mask, temperatures, class_name, class_color, tower_angle=0.0):
  class_mask = extract_class_mask(mask, class_color)
  if np.sum(class_mask) == 0:
    return []

  labeled, num_components = label_connected_components(class_mask)
  if num_components == 0:
    return []

  # 1. Extract individual component masks
  raw_masks = []
  for c_id in range(1, num_components + 1):
    comp_mask = labeled == c_id
    if np.sum(comp_mask) >= 15:  # Filter noise
      raw_masks.append(comp_mask)

  if not raw_masks:
    return []

  # 2. Group masks that are within 'MAX_GAP' pixels of each other
  MAX_GAP = 12  # Distance in pixels across the intersecting wire
  merged_component_masks = []
  visited = [False] * len(raw_masks)

  for i in range(len(raw_masks)):
    if visited[i]:
      continue

    # Start a new combined component group with exact original pixels
    combined_mask = raw_masks[i].copy()
    visited[i] = True

    for j in range(i + 1, len(raw_masks)):
      if visited[j]:
        continue

      # Check shortest pixel distance between mask i and mask j
      dist_map = distance_transform_edt(~combined_mask)
      min_distance = np.min(dist_map[raw_masks[j]])

      if min_distance <= MAX_GAP:
        combined_mask |= raw_masks[j]  # Logical OR combines both sides
        visited[j] = True

    merged_component_masks.append(combined_mask)

  # 3. Analyze each combined component group
  results = []
  for idx, comp_mask in enumerate(merged_component_masks, start=1):
    print(f'      Component #{idx}: ', end='')

    # stats = analyze_component_simple(
    #     comp_mask, temperatures, idx, class_name
    # )

    stats = analyze_component_simple(
    comp_mask, temperatures, idx, class_name, tower_angle
    )


    if stats:
      results.append(stats)
      print(f"✓ {stats['max_temp']:.1f}°C [{stats['orientation']}]")
    else:
      print('✗ Failed')

  return results
def print_analysis_results(all_results):
    """Print detailed analysis results including orientation"""
    print("\n" + "="*90)
    print("THERMAL COMPONENT ANALYSIS - ACCURATE TEMPERATURES (DJI SDK)")
    print("="*90)
    
    for class_name in ['insulator_up', 'insulator_down', 'wires', 'tower_structure', 'clamps', 'fixer']:
        class_results = [r for r in all_results if r['class'] == class_name]
        
        if not class_results:
            print(f"\n{class_name.upper()}: No components found")
            continue
        
        print(f"\n{class_name.upper()}")
        print("-" * 90)
        
        for result in class_results:
            print(f"  Component #{result['component_id']}:")
            print(f"    Orientation:   {result['orientation']}")
            print(f"    Max Temp:     {result['max_temp']:7.2f}°C  at pixel ({result['max_location'][0]:4d}, {result['max_location'][1]:4d})")
            print(f"    Min Temp:     {result['min_temp']:7.2f}°C")
            print(f"    Mean Temp:    {result['mean_temp']:7.2f}°C")
            print(f"    Median Temp:  {result['median_temp']:7.2f}°C")
            print(f"    Std Dev:      {result['std_dev']:7.2f}°C")
            print(f"    Pixels:       {result['pixel_count']:,}")
            print()


def get_decimal_from_dms(dms, ref):
    """Converts EXIF degrees/minutes/seconds to decimal format."""
    degrees = float(dms[0].num) / float(dms[0].den)
    minutes = float(dms[1].num) / float(dms[1].den)
    seconds = float(dms[2].num) / float(dms[2].den)

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal


def print_location_on_image(image_path):
    # 1. Read EXIF metadata from thermal file
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

    lat = tags.get('GPS GPSLatitude')
    lat_ref = tags.get('GPS GPSLatitudeRef')
    lon = tags.get('GPS GPSLongitude')
    lon_ref = tags.get('GPS GPSLongitudeRef')

    if not (lat and lat_ref and lon and lon_ref):
        print("No GPS metadata found in this image.")
        return

    # 2. Format location string
    lat_dec = get_decimal_from_dms(lat.values, lat_ref.printable)
    lon_dec = get_decimal_from_dms(lon.values, lon_ref.printable)
    location_text = f"GPS: {lat_dec:.5f}, {lon_dec:.5f}"

    return location_text

# def create_marked_visualization(temperatures, mask, all_results, location_info):
#     """Create visualization with marked hotspots"""
#     # Normalize temperatures for display
#     temp_normalized = cv2.normalize(temperatures.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     temp_colored = cv2.applyColorMap(temp_normalized, cv2.COLORMAP_TURBO)
    
#     marked_image = temp_colored.copy()
    
#     # Mark each component's max temperature
#     for result in all_results:
#         class_name = result['class']
#         x, y = result['max_location']
#         max_temp = result['max_temp']
#         component_id = result['component_id']
        
#         # color = marking_colors.get(class_name, (255, 255, 255))
#         color = (0, 0, 0)
        
#         # Draw circle and crosshair
#         cv2.circle(marked_image, (x, y), radius=15, color=color, thickness=1)
#         cv2.line(marked_image, (x - 30, y), (x + 30, y), color, 1)
#         cv2.line(marked_image, (x, y - 30), (x, y + 30), color, 1)
        
#         # Label
#         # label = f"{class_name[:8]}#{component_id}\n{max_temp:.1f}C"
#         label = f"{class_name[:8]}:\n{max_temp:.1f}C"
        
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.6
#         thickness = 2
        
#         text_size = cv2.getTextSize(label.split('\n')[0], font, font_scale, thickness)[0]
#         text_x = x + 35
#         text_y = y
        
#         # # Draw background
#         # cv2.rectangle(marked_image, 
#         #              (text_x - 5, text_y - text_size[1] - 10),
#         #              (text_x + text_size[0] + 5, text_y + 25),
#         #              (0, 0, 0), -1)

#         # Draws an outlined bounding box around the text (thickness = 2)
#         # cv2.rectangle(marked_image, 
#         #             (text_x - 5, text_y - text_size[1] - 5),
#         #             (text_x + text_size[0] + 5, text_y + 5),
#         #             (0, 0, 0), 2)  # Change (0, 255, 0) to your desired BGR color
                
#         # Draw text
#         cv2.putText(marked_image, label.replace('\n', ' '), 
#                    (text_x, text_y), font, font_scale, color, thickness)

#         text_position = (20, 40)  # (X, Y) top-left text origin
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.7
#         padding = 6  # Space around the text in pixels

#         # 2. Get text width, height, and baseline
#         (text_w, text_h), baseline = cv2.getTextSize(
#             location_info, font, font_scale, thickness=2
#         )

#         # 3. Calculate bounding box coordinates
#         box_x1 = text_position[0] - padding
#         box_y1 = text_position[1] - text_h - padding
#         box_x2 = text_position[0] + text_w + padding
#         box_y2 = text_position[1] + baseline + padding

#         # 4. Draw background box (Dark grey/black filled rectangle)
#         bg_color = (0, 0, 0)  # Change BGR color if desired (e.g., (50, 50, 50))
#         cv2.rectangle(
#             marked_image, (box_x1, box_y1), (box_x2, box_y2), bg_color, thickness=-1
#         )

#         # 5. Draw outline and text on top of the background
#         cv2.putText(
#             marked_image,
#             location_info,
#             text_position,
#             font,
#             font_scale,
#             (0, 0, 0),
#             4,
#         )  # Black stroke outline
#         cv2.putText(
#             marked_image,
#             location_info,
#             text_position,
#             font,
#             font_scale,
#             (0, 255, 255),
#             2,
#         )
    
#     return marked_image




def create_marked_visualization(temperatures,mask,all_results,location_info):

    temp_normalized = cv2.normalize(
        temperatures.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    temp_colored = cv2.applyColorMap(
        temp_normalized,
        cv2.COLORMAP_TURBO
    )

    marked_image = temp_colored.copy()

    H, W = marked_image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    used_label_boxes = []

    def boxes_overlap(box1, box2, margin=10):
        """Check if two boxes overlap with a margin."""
        
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2

        return not (
            x2 + margin < a1 or
            a2 + margin < x1 or
            y2 + margin < b1 or
            b2 + margin < y1
        )

    def get_label_box(text,center,font_scale=0.50,thickness=2):
        """Calculate bounding box for a label."""

        (text_w, text_h), baseline = cv2.getTextSize(text,font,font_scale,thickness)

        padding_x = 7
        padding_y = 5

        x1 = int(center[0] - text_w / 2 - padding_x)
        y1 = int(center[1] - text_h - padding_y)
        x2 = int(center[0] + text_w / 2 + padding_x)
        y2 = int(center[1] + baseline + padding_y)

        return (x1, y1, x2, y2)

    def draw_label(
        image,
        text,
        center,
        font_scale=0.50,
        text_color=(255, 255, 255)
    ):
        """Draw label with background, ensuring it stays in bounds."""

        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        padding_x = 7
        padding_y = 5

        x1 = int(center[0] - text_w / 2 - padding_x)
        y1 = int(center[1] - text_h - padding_y)
        x2 = int(center[0] + text_w / 2 + padding_x)
        y2 = int(center[1] + baseline + padding_y)


        x1 = max(5, x1)
        y1 = max(5, y1)
        x2 = min(W - 5, x2)
        y2 = min(H - 5, y2)

        # Skip if box became too small
        if (x2 - x1) < 20 or (y2 - y1) < 10:
            return (x1, y1, x2, y2)

        # --------------------------------------------------------
        # Semi-transparent black background
        # --------------------------------------------------------

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.75,
            image,
            0.25,
            0,
            image
        )

        # --------------------------------------------------------
        # Thin border
        # --------------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (180, 180, 180),
            1
        )

        text_x = int((x1 + x2 - text_w) / 2)
        text_y = int((y1 + y2 + text_h) / 2)

        # Clamp text position to box
        text_x = max(x1 + 2, min(x2 - text_w - 2, text_x))
        text_y = max(y1 + text_h, min(y2 - 2, text_y))

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

        return (x1, y1, x2, y2)

    # ============================================================
    # 7. HELPER: FIND NON-OVERLAPPING LABEL POSITION
    # ============================================================

    def find_label_position(
        hot_x,
        hot_y,
        text,
        font_scale=0.50
    ):
        """
        IMPROVED: Find non-overlapping label position with adaptive font sizing.
        
        Returns: (position, box, adaptive_font_scale)
        """

        # ====================================================
        # TIER 1: PRIMARY CANDIDATES WITH NORMAL FONT
        # ====================================================

        candidates = [
            # Primary positions (highest priority)
            (hot_x, hot_y - 75),              # Above
            (hot_x + 110, hot_y - 65),        # Upper-right
            (hot_x - 110, hot_y - 65),        # Upper-left
            (hot_x + 140, hot_y),             # Right
            (hot_x - 140, hot_y),             # Left
            
            # Secondary positions
            (hot_x + 90, hot_y + 70),         # Lower-right
            (hot_x - 90, hot_y + 70),         # Lower-left
            (hot_x, hot_y + 90),              # Below
            
            # Tertiary positions (far away)
            (hot_x, hot_y - 120),             # Far above
            (hot_x + 160, hot_y - 90),        # Far upper-right
            (hot_x - 160, hot_y - 90),        # Far upper-left
            (hot_x, hot_y + 130),             # Far below
            
            # Corners (last resort)
            (hot_x + 150, hot_y + 100),       # Lower-right corner
            (hot_x - 150, hot_y + 100),       # Lower-left corner
        ]

        current_font_scale = font_scale

        for candidate in candidates:
            box = get_label_box(text, candidate, current_font_scale)
            x1, y1, x2, y2 = box

            # Boundary check
            if x1 < 5 or y1 < 5 or x2 >= W - 5 or y2 >= H - 5:
                continue

            # Overlap check
            collision = False
            for existing_box in used_label_boxes:
                if boxes_overlap(box, existing_box, margin=15):
                    collision = True
                    break

            if not collision:
                return candidate, box, current_font_scale

        # ====================================================
        # TIER 2: SAME POSITIONS WITH SMALLER FONT
        # ====================================================

        smaller_font_scale = font_scale * 0.75
        current_font_scale = smaller_font_scale

        for candidate in candidates:
            box = get_label_box(text, candidate, current_font_scale)
            x1, y1, x2, y2 = box

            if x1 < 5 or y1 < 5 or x2 >= W - 5 or y2 >= H - 5:
                continue

            collision = False
            for existing_box in used_label_boxes:
                if boxes_overlap(box, existing_box, margin=12):
                    collision = True
                    break

            if not collision:
                return candidate, box, current_font_scale

        # ====================================================
        # TIER 3: GRID-BASED SEARCH WITH TINY FONT
        # ====================================================

        tiny_font_scale = font_scale * 0.6
        current_font_scale = tiny_font_scale

        grid_positions = []
        for dx in range(-200, 220, 50):
            for dy in range(-150, 170, 50):
                if dx == 0 and dy == 0:
                    continue
                grid_positions.append((hot_x + dx, hot_y + dy))

        for candidate in grid_positions:
            box = get_label_box(text, candidate, current_font_scale)
            x1, y1, x2, y2 = box

            if x1 < 5 or y1 < 5 or x2 >= W - 5 or y2 >= H - 5:
                continue

            collision = False
            for existing_box in used_label_boxes:
                if boxes_overlap(box, existing_box, margin=10):
                    collision = True
                    break

            if not collision:
                return candidate, box, current_font_scale


        ultra_tiny_font = font_scale * 0.5
        current_font_scale = ultra_tiny_font

        corner_positions = [
            (50, 50),                         # Top-left
            (W - 150, 50),                    # Top-right
            (50, H - 50),                     # Bottom-left
            (W - 150, H - 50),                # Bottom-right
            (W // 2, 50),                     # Top-center
            (W // 2, H - 50),                 # Bottom-center
        ]

        for candidate in corner_positions:
            box = get_label_box(text, candidate, current_font_scale)
            x1, y1, x2, y2 = box

            if x1 < 5 or y1 < 5 or x2 >= W - 5 or y2 >= H - 5:
                continue

            return candidate, box, current_font_scale

        # ====================================================
        # TIER 5: LAST RESORT - TRUNCATE & PLACE SAFELY
        # ====================================================

        truncated_text = text[:10] + "..." if len(text) > 10 else text
        current_font_scale = 0.4

        safe_candidate = (
            max(50, min(W - 80, hot_x)),
            max(50, min(H - 30, hot_y - 60))
        )

        box = get_label_box(truncated_text, safe_candidate, current_font_scale)

        return safe_candidate, box, current_font_scale

    # ============================================================
    # 8. PROCESS EVERY COMPONENT
    # ============================================================

    for result in all_results:

        class_name = result['class']
        x, y = result['max_location']
        x = int(x)
        y = int(y)

        max_temp = result['max_temp']
        component_id = result['component_id']
        orientation = result.get('orientation', '')

        if class_name == 'insulator_up':
            label = (
                    f"#{component_id}  "
                    f"Vertival  "
                    f"{max_temp:.1f}°C"
                )

            cv2.circle(
            marked_image,
            (x, y),
            3,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
            )

            # Small dark outline
            cv2.circle(
                marked_image,
                (x, y),
                4,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )


            label_center, label_box, adaptive_font = find_label_position(
                x,
                y,
                label,
                font_scale=0.50
            )

            draw_label(
                marked_image,
                label,
                label_center,
                font_scale=adaptive_font
            )

            used_label_boxes.append(label_box)


            bx1, by1, bx2, by2 = label_box
            label_center_x = int((bx1 + bx2) / 2)
            label_center_y = int((by1 + by2) / 2)

            distance = np.hypot(label_center_x - x, label_center_y - y)

            if distance > 65:

                # Determine nearest edge of label
                if label_center_y < y:
                    connection_point = (label_center_x, by2)
                elif label_center_y > y:
                    connection_point = (label_center_x, by1)
                elif label_center_x > x:
                    connection_point = (bx1, label_center_y)
                else:
                    connection_point = (bx2, label_center_y)

                # Draw line
                cv2.line(
                    marked_image,
                    (x, y),
                    connection_point,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.line(
                    marked_image,
                    (x, y),
                    connection_point,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        elif class_name == 'insulator_down':
            label = (
                    f"#{component_id}  "
                    f"Horizontal  "
                    f"{max_temp:.1f}°C"
                )

            cv2.circle(marked_image,(x, y),3,(255, 255, 255),-1,cv2.LINE_AA)

            # Small dark outline
            cv2.circle(marked_image,(x, y),4,(0, 0, 0),1,cv2.LINE_AA)


            label_center, label_box, adaptive_font = find_label_position(x,y,label,font_scale=0.50)

            draw_label(marked_image,label,label_center,font_scale=adaptive_font)

            used_label_boxes.append(label_box)


            bx1, by1, bx2, by2 = label_box
            label_center_x = int((bx1 + bx2) / 2)
            label_center_y = int((by1 + by2) / 2)

            distance = np.hypot(label_center_x - x, label_center_y - y)

            if distance > 65:

                # Determine nearest edge of label
                if label_center_y < y:
                    connection_point = (label_center_x, by2)
                elif label_center_y > y:
                    connection_point = (label_center_x, by1)
                elif label_center_x > x:
                    connection_point = (bx1, label_center_y)
                else:
                    connection_point = (bx2, label_center_y)

                # Draw line
                cv2.line(marked_image,(x, y),connection_point,(0, 0, 0),2,cv2.LINE_AA)

                cv2.line(marked_image,(x, y),connection_point,(255, 255, 255),1,cv2.LINE_AA)



        # ========================================================
        # ALL OTHER COMPONENTS
        # ========================================================

        else:

            color = (0, 0, 0)
            # Original hotspot circle
            cv2.circle(marked_image,(x, y),radius=15,color=color,thickness=1)

            # Original crosshair
            cv2.line(marked_image,(x - 30, y),(x + 30, y),color,1)

            cv2.line(marked_image,(x, y - 30),(x, y + 30),color,1 )

            # --------------------------------------------------------
            # LABEL WITH IMPROVED POSITIONING
            # --------------------------------------------------------

            label = (
                f"{class_name[:8]}:"
                f"{max_temp:.1f}C"
            )

            # Use improved positioning for all components
            label_center, label_box, adaptive_font = find_label_position(
                x,
                y,
                label,
                font_scale=0.50
            )

            draw_label(
                marked_image,
                label,
                label_center,
                font_scale=adaptive_font
            )

            used_label_boxes.append(label_box)

            # --------------------------------------------------------
            # LEADER LINE (from hotspot to label)
            # Same as insulators
            # --------------------------------------------------------

            bx1, by1, bx2, by2 = label_box
            label_center_x = int((bx1 + bx2) / 2)
            label_center_y = int((by1 + by2) / 2)

            distance = np.hypot(label_center_x - x, label_center_y - y)

            if distance > 65:

                # Determine nearest edge of label
                if label_center_y < y:
                    connection_point = (label_center_x, by2)
                elif label_center_y > y:
                    connection_point = (label_center_x, by1)
                elif label_center_x > x:
                    connection_point = (bx1, label_center_y)
                else:
                    connection_point = (bx2, label_center_y)

                # Draw line (black outline)
                cv2.line(
                    marked_image,
                    (x, y),
                    connection_point,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                # Draw line (white highlight)
                cv2.line(
                    marked_image,
                    (x, y),
                    connection_point,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

    # ============================================================
    # 9. LOCATION INFORMATION
    # ============================================================

    if location_info:

        text_position = (20, 40)
        font_scale = 0.7
        thickness = 2
        padding = 6

        (text_w, text_h), baseline = cv2.getTextSize(
            location_info,
            font,
            font_scale,
            thickness
        )

        box_x1 = text_position[0] - padding
        box_y1 = text_position[1] - text_h - padding
        box_x2 = text_position[0] + text_w + padding
        box_y2 = text_position[1] + baseline + padding

        # Clamp to image bounds
        box_x1 = max(2, box_x1)
        box_y1 = max(2, box_y1)
        box_x2 = min(W - 2, box_x2)
        box_y2 = min(H - 2, box_y2)

        # Background
        overlay = marked_image.copy()

        cv2.rectangle(
            overlay,
            (box_x1, box_y1),
            (box_x2, box_y2),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.70,
            marked_image,
            0.30,
            0,
            marked_image
        )

        # Location text with black outline
        cv2.putText(
            marked_image,
            location_info,
            text_position,
            font,
            font_scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )

        # Yellow text
        cv2.putText(
            marked_image,
            location_info,
            text_position,
            font,
            font_scale,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ============================================================
    # 10. RETURN
    # ============================================================

    return marked_image
def create_comprehensive_report(temperatures, mask, marked_image, all_results):
    """Create comprehensive analysis report"""
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Temperature heatmap
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(temperatures, cmap='hot')
    ax1.set_title('Temperature Heatmap (°C)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
    ax1.axis('off')
    
    # Plot 2: Marked image
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('Marked Hotspots', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Plot 3: Segmentation mask
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    ax3.set_title('Segmentation Mask', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Plot 4: Temperature histogram
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(temperatures.flatten(), bins=100, color='red', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Temperature (°C)', fontsize=10)
    ax4.set_ylabel('Pixel Count', fontsize=10)
    ax4.set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Max temps by class
    ax5 = plt.subplot(3, 3, 5)
    class_max_temps = defaultdict(list)
    for result in all_results:
        class_max_temps[result['class']].append(result['max_temp'])
    
    if class_max_temps:
        classes = list(class_max_temps.keys())
        max_temps = [max(temps) for temps in class_max_temps.values()]
        colors_list = [marking_colors.get(c, (255, 255, 255)) for c in classes]
        colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors_list]
        
        bars = ax5.bar(range(len(classes)), max_temps, color=colors_rgb, edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(classes)))
        ax5.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax5.set_ylabel('Temperature (°C)', fontsize=10)
        ax5.set_title('Max Temperature per Class', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, temp in zip(bars, max_temps):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{temp:.1f}°C', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Mean temps by class
    ax6 = plt.subplot(3, 3, 6)
    class_mean_temps = defaultdict(list)
    for result in all_results:
        class_mean_temps[result['class']].append(result['mean_temp'])
    
    if class_mean_temps:
        classes = list(class_mean_temps.keys())
        mean_temps = [np.mean(temps) for temps in class_mean_temps.values()]
        colors_list = [marking_colors.get(c, (255, 255, 255)) for c in classes]
        colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors_list]
        
        bars = ax6.bar(range(len(classes)), mean_temps, color=colors_rgb, edgecolor='black', linewidth=1.5)
        ax6.set_xticks(range(len(classes)))
        ax6.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax6.set_ylabel('Temperature (°C)', fontsize=10)
        ax6.set_title('Mean Temperature per Class', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, temp in zip(bars, mean_temps):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{temp:.1f}°C', ha='center', va='bottom', fontsize=8)
    
    # Plot 7-9: Statistics table (split into 3 tables)
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    table_data = [['Class', 'ID', 'Max (°C)', 'Mean (°C)', 'Min (°C)']]
    for result in sorted(all_results, key=lambda x: (-x['max_temp'], x['class'])):
        table_data.append([
            result['class'][:10],
            str(result['component_id']),
            f"{result['max_temp']:.1f}",
            f"{result['mean_temp']:.1f}",
            f"{result['min_temp']:.1f}"
        ])
    
    if len(table_data) > 1:
        table = ax7.table(cellText=table_data[:6], cellLoc='center', loc='center',
                         colWidths=[0.2, 0.15, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax7.set_title('Temperature Statistics (Top Results)', fontsize=11, fontweight='bold', pad=10)
    
    # Summary boxes
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    if all_results:
        max_temp_overall = max(r['max_temp'] for r in all_results)
        min_temp_overall = min(r['min_temp'] for r in all_results)
        mean_temp_overall = np.mean([r['mean_temp'] for r in all_results])
        
        summary_text = f"""
        OVERALL STATISTICS
        ━━━━━━━━━━━━━━━━━━━
        
        Max Temperature:  {max_temp_overall:.2f}°C
        Min Temperature:  {min_temp_overall:.2f}°C
        Mean Temperature: {mean_temp_overall:.2f}°C
        
        Components:       {len(all_results)}
        
        Temperature Range: {max_temp_overall - min_temp_overall:.2f}°C
        """
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    hotspot_text = f"""
    HOTSPOT ANALYSIS
    ━━━━━━━━━━━━━━━━━━━
    
    """
    
    hottest = sorted(all_results, key=lambda x: x['max_temp'], reverse=True)[:5]
    for i, result in enumerate(hottest, 1):
        hotspot_text += f"{i}. {result['class'][:8]} #{result['component_id']}\n"
        hotspot_text += f"   {result['max_temp']:.1f}°C\n\n"
    
    ax9.text(0.1, 0.9, hotspot_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    try:
        print("="*90)
        print("DJI THERMAL IMAGE ANALYSIS - INSULATOR ORIENTATION DETECTION")
        print("="*90)
        
        # Step 1: Extract temperatures
        print("\n[Step 1] Extracting thermal data...")
        temperatures = extract_dji_thermal_accurate(INPUT_THERMAL, DLL_PATH)
        print(f"✓ Temperature range: {temperatures.min():.2f}°C to {temperatures.max():.2f}°C")
        
        # Step 2: Load mask
        print("\n[Step 2] Loading segmentation mask...")
        mask = cv2.imread(INPUT_MASK)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {INPUT_MASK}")
        
        # # Step 3: Extract depth
        # print("\n[Step 3] Extracting depth map...")
        raw_image = cv2.imread(INPUT_THERMAL)

        # Compute reference tower orientation angle
        tower_angle = get_tower_orientation(mask)
        # if abs(tower_angle) > 70:
        #     tower_angle = 90.0


        print(f"✓ Extracted Reference Tower Angle: {tower_angle:.2f}°")


        # depth_raw = depth_anything.infer_image(raw_image, 1022)
        # depth_normalized = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)
        # print(f"✓ Depth range: {depth_normalized.min():.4f} to {depth_normalized.max():.4f}")
        
        # Step 4: Analyze with NEW refinement
        # print("\n[Step 4] Analyzing components with erosion + depth surface detection...")
        all_results = []

        for class_name, class_color in class_colors.items():
            if class_name == 'background':
                continue
            
            print(f"  {class_name:20s}:")
            results = process_class(mask, temperatures, class_name, class_color, tower_angle)
            all_results.extend(results)
        
        print_analysis_results(all_results)
        
        # # Step 5: Visualize
        # print("\n[Step 5] Creating visualizations...")
        location_info = print_location_on_image(INPUT_THERMAL)
        # print(location_info)

    
        marked_image = create_marked_visualization(raw_image, mask, all_results, location_info)


        # cv2.imwrite(r'D:\transmition-line inspection\Depth-Anything-V2\thermal_marked_erosion_depth3.jpg', marked_image)
        cv2.imwrite(directory+"//"+ image_name + "_analysed.jpg", marked_image)
        print("✓ Saved: thermal_marked_erosion_depth.jpg")
        
        print("\n" + "="*90)
        print("ANALYSIS COMPLETE")
        print("="*90)
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()