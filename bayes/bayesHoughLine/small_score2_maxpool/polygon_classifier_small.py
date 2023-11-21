import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import argparse
import cv2
import glob
import skimage.measure
import laspy
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, mapping
import shapely
from matplotlib.path import Path
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
import os


class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, path="~/data/", canny_lower=4, canny_upper=160,hough_lines_treshold=10,
                  min_line_length=5,max_line_gap=10, closing_kernel_size=5, opening_kernel_size=3, small_dialation_kernel=3,
                    meters_around_line=1, simplify_tolerance=1, cc_area=50, distance_resolution=1):
        
        self.path = path
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper
        self.hough_lines_treshold = hough_lines_treshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.closing_kernel_size = closing_kernel_size
        self.opening_kernel_size = opening_kernel_size
        self.small_dialation_kernel=small_dialation_kernel
        self.meters_around_line = meters_around_line
        self.simplify_tolerance = simplify_tolerance
        self.cc_area = cc_area
        self.distance_resolution = distance_resolution
        self.indexes_needed_list = []        

    def GenPath(self, path):
        if path[-1] == '/':
            return path
        else:
            return path+'/'

    def GetPathRelations(self):
        full_path_to_data = self.GenPath(self.path)
        
        ground_removed_image_paths = []
        laz_point_cloud_paths = []
        
        # Find full path to all images
        for path in glob.glob(full_path_to_data+'ImagesGroundRemovedSmall/*'):
            ground_removed_image_paths.append(path)
    
        # Find full path to all laz files
        for path in glob.glob(full_path_to_data+'LazFilesWithHeightRemoved/*'):
            laz_point_cloud_paths.append(path)
            
        ground_removed_image_paths.sort()
        laz_point_cloud_paths.sort()
        assert(len(ground_removed_image_paths)==len(laz_point_cloud_paths))
        return ground_removed_image_paths, laz_point_cloud_paths

    def BBTouchingEdge(self, image_shape, bb, epsilon):
        image_width, image_height = image_shape
        left, top, width, height = bb[0], bb[1], bb[2], bb[3]
        right = left + width
        bottom = top + height
        distance_to_left = left
        distance_to_right = image_width - right
        distance_to_top = top
        distance_to_bottom = image_height - bottom

        if distance_to_left > epsilon and distance_to_right > epsilon and distance_to_top > epsilon and distance_to_bottom > epsilon:
            return False
        else:
            return True

    def hough_lines(self, file: str):

        # Create Image
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = np.where(image >= 0, image, 0)
        image = image/np.max(image)
        image = (image*255).astype(np.uint8)

        image = skimage.measure.block_reduce(image, (3,3), np.max)

        # Apply Closing
        closing_kernel = np.ones((self.closing_kernel_size,self.closing_kernel_size),np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, closing_kernel)

        # Apply Closing
        opening_kernel = np.ones((self.opening_kernel_size,self.opening_kernel_size),np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, opening_kernel)

        # Apply Edge Detection
        edges = cv2.Canny(opening, self.canny_lower, self.canny_upper, None, 3)

        # Apply HoughLinesP
        linesP = cv2.HoughLinesP(
            edges, # Input edge image
            self.distance_resolution, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=self.hough_lines_treshold, # Min number of votes for valid line
            minLineLength=self.min_line_length, # Min allowed length of line
            maxLineGap=self.max_line_gap # Max allowed gap between line for joining them
            )
        
        lines_image = np.zeros_like(edges)
        # Draw the lines
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(lines_image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3)


        ## Small Dilation Kernel
        kernel = np.ones((self.small_dialation_kernel, self.small_dialation_kernel), np.uint8)
        lines_image = cv2.dilate(lines_image, kernel, iterations=1)

        (_, label_ids, bounding_box, _) = cv2.connectedComponentsWithStats(lines_image)
        for i in range(len(bounding_box)):
            # Must be 10 Pixels from the edge of the image
            if not self.BBTouchingEdge(image.shape, bounding_box[i], 3):
                area = bounding_box[i][cv2.CC_STAT_AREA]
                if area < self.cc_area:
                    lines_image[label_ids == i] = 0

        # Pixels per kilometer
        x_pixels, y_pixels = image.shape

        # Pixels per meter
        x_per_km_pixels, y_per_km_pixels = x_pixels/1000, y_pixels/1000

        kernel_size = int(self.meters_around_line*np.ceil(x_per_km_pixels))

        # Apply Dilation and create a cirkular kernel using (image, center_coordinates, radius, color, thickness)
        circular_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cv2.circle(circular_kernel, (int(kernel_size/2), int(kernel_size/2)), int(kernel_size/2), 255, -1)
        dilation_cirkular_kernel = cv2.dilate(lines_image, circular_kernel, iterations=1)
        return dilation_cirkular_kernel, image


    def make_polygons(self, dilation_cirkular_kernel):
        # Create Polygons and Multi Polygons
        mask = (dilation_cirkular_kernel == 255)
        output = rasterio.features.shapes(dilation_cirkular_kernel, mask=mask, connectivity=4)
        output_list = list(output)

        all_polygons = []
        all_multi_polygons =[]

        for multi_polygon in output_list:
            found_polygon = multi_polygon[0]['coordinates']
            # Then its just a Polygon
            if len(found_polygon) == 1:
                all_polygons.append(Polygon(found_polygon[0]))
            # Else its a multipolygon
            else:
                tmpMulti = []
                for p in found_polygon:
                    tmpMulti.append(Polygon(p))
                all_multi_polygons.append(tmpMulti)


        # Remove all low area multipolygons
        for i, multi_pol in enumerate(all_multi_polygons):
            new_list = [multi_pol[0]]
            # No matter what, dont remove the first one
            for pol in multi_pol[1:]:
                new_list.append(pol)
            all_multi_polygons[i] = new_list

        simplified_all_polygons = []
        simplified_all_multi_polygons =[]
        # Simplify all standard polygons
        for p in all_polygons:
            simplified_all_polygons.append(shapely.simplify(p, tolerance=self.simplify_tolerance, preserve_topology=True))
        simplified_all_polygons  = [p for p in simplified_all_polygons if not p.is_empty]

        # Simplify all multi polygons
        for multi_pol in all_multi_polygons:
            tmp = []
            for p in multi_pol:
                tmp.append(shapely.simplify(p, tolerance=self.simplify_tolerance, preserve_topology=True))
            tmp  = [p for p in tmp if not p.is_empty]
            simplified_all_multi_polygons.append(tmp)

        # Create bounding box polygons
        bbox_all_polygon_path = []
        tmp = [p.bounds for p in simplified_all_polygons]
        for values in tmp:
            #values = (minx, miny, maxx, maxy)
            x_min = values[0]
            x_max = values[2]
            y_min = values[1]
            y_max = values[3]
            bb = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
            bbox_all_polygon_path.append(Path(bb))

        # Create bounding box for multi polygons
        bbox_all_multi_polygons_path = []
        for multi_pol in simplified_all_multi_polygons:
            tmp = [p.bounds for p in multi_pol]
            tmp_multi_pol_boxes = []

            for values in tmp:
                #values = (minx, miny, maxx, maxy)
                x_min = values[0]
                x_max = values[2]
                y_min = values[1]
                y_max = values[3]
                bb = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
                tmp_multi_pol_boxes.append(Path(bb))
            bbox_all_multi_polygons_path.append(tmp_multi_pol_boxes)

        # Create Path polygons from the simplified shapely polygons
        simplified_all_polygons_path = [Path(mapping(p)['coordinates'][0]) for p in simplified_all_polygons]
        simplified_all_multi_polygons_path = []
        for multi_pol in simplified_all_multi_polygons:
            tmp = [Path(mapping(p)['coordinates'][0]) for p in multi_pol]
            simplified_all_multi_polygons_path.append(tmp)

        return simplified_all_polygons_path, simplified_all_multi_polygons_path, bbox_all_polygon_path, bbox_all_multi_polygons_path

    def MaxMinNormalize(self, arr):
        return (arr - np.min(arr))/(np.max(arr)-np.min(arr))

    def CastAllXValuesToImage(self, arr, x_pixels):
        return self.MaxMinNormalize(arr)*x_pixels

    def CastAllYValuesToImage(self, arr, y_pixels):
        return (1-self.MaxMinNormalize(arr))*y_pixels

    def filter_polygons(self, reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image):
        # Pixels per kilometer
        x_pixels, y_pixels = image.shape
        x_values = np.rint(self.CastAllXValuesToImage(point_cloud.X, x_pixels))
        y_values = np.rint(self.CastAllYValuesToImage(point_cloud.Y, y_pixels))

        move_in = 2
        x_values = np.where(x_values < x_pixels, x_values, x_pixels-move_in)
        x_values = np.where(x_values >= 0, x_values, 0+move_in)
        
        y_values = np.where(y_values < y_pixels, y_values, y_pixels-move_in)
        y_values = np.where(y_values >= 0, y_values, 0+move_in)

        # Format: [(1,1), (3,5), (1,5), ...] with 30 mio samples
        list_zipped = np.array(list(zip(x_values, y_values)))
              
        # Generate a bool list to obtain the final indexes from the dataset
        indexes_needed = np.zeros(len(x_values), dtype=bool)

        # Run through all polygons and check which points are inside the polygon
        for i in range(len(reg_polygons)):
            # Check if point is inside the bounding box
            indexes_inside_box = bbox_reg_polygon[i].contains_points(list_zipped)
            indexes_inside_box = np.array([index for index, x in enumerate(indexes_inside_box) if x])
            
            # Generate small dataset
            if len(indexes_inside_box) != 0:
                tmp = list_zipped[indexes_inside_box]
                
                # Check if any of these points are in the polygon
                indexes_inside_polygon = reg_polygons[i].contains_points(tmp)
                
                # Find the indexes from the box that is also inside the polygon
                final_indexes = indexes_inside_box[indexes_inside_polygon]
                
                # Update the indexes
                indexes_needed[final_indexes] = 1

        for i in range(len(multi_polygons)):
            tmp_indexes_needed = np.zeros(len(x_values), dtype=bool)
            tmp_indexes_not_needed = np.zeros(len(x_values), dtype=bool)
            
            # Get the current bb multipolygon and the current simplified multipolygon
            bb_multi_pol = bbox_multi_polygons[i]
            simpli_multi_pol = multi_polygons[i]
            
            # Find the indexes that are inside the bounding box of the first element
            indexes_inside_box = bb_multi_pol[0].contains_points(list_zipped)
            indexes_inside_box = np.array([index for index, x in enumerate(indexes_inside_box) if x])
            
            # Generate smaller dataset
            tmp = list_zipped[indexes_inside_box]
            
            # Check if any of these points are in the polygon
            indexes_inside_polygon = simpli_multi_pol[0].contains_points(tmp)
            
            # Find the indexes from the box that is also inside the polygon
            final_indexes = indexes_inside_box[indexes_inside_polygon]
            tmp_indexes_needed[final_indexes] = 1
                
            for j in range(1, len(bb_multi_pol)):
                
                # Get the bounding box of the temp multi polygon
                indexes_inside_box = bb_multi_pol[j].contains_points(list_zipped)
                indexes_inside_box = np.array([index for index, x in enumerate(indexes_inside_box) if x])
                
                # Generate small dataset
                if len(indexes_inside_box) != 0:
                    tmp = list_zipped[indexes_inside_box]
                
                    # Check if any of these points are in the polygon
                    indexes_inside_polygon = simpli_multi_pol[j].contains_points(tmp)
                    final_indexes = indexes_inside_box[indexes_inside_polygon]
                    
                    # Update the indexes
                    tmp_indexes_not_needed[final_indexes] = 1
        
                    indexes_needed = indexes_needed | (tmp_indexes_needed & np.invert(tmp_indexes_not_needed))

        return indexes_needed

    def fit(self, X, y):
        ground_removed_image_paths, laz_point_cloud_paths = self.GetPathRelations()
        for tif, laz_file in zip(ground_removed_image_paths, laz_point_cloud_paths):
            dilation_cirkular_kernel, image = self.hough_lines(tif)
            reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, = self.make_polygons(dilation_cirkular_kernel)
            
            point_cloud = laspy.read(laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)
            indexes_needed = self.filter_polygons(reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image)
            self.indexes_needed_list.append(indexes_needed)

    def score(self, _, __):
        _, laz_point_cloud_paths = self.GetPathRelations()

        pct_lost_datapoints_list = []
        pct_kept_powerline_list = []

        for i, laz_file in enumerate(laz_point_cloud_paths):
            point_cloud = laspy.read(laz_file, laz_backend=laspy.compression.LazBackend.LazrsParallel)

            indexes_needed = self.indexes_needed_list[i]
            new_point_data = point_cloud[indexes_needed]

            amount_wire = np.sum(point_cloud.classification == 14)
            new_amount_wire = np.sum(new_point_data.classification == 14)

            pct_kept_powerline = new_amount_wire/amount_wire
            pct_kept_powerline_list.append(pct_kept_powerline)

            amount_points = len(point_cloud)
            new_amount_points = len(new_point_data)

            pct_lost_datapoints = 1-(new_amount_points/amount_points)
            pct_lost_datapoints_list.append(pct_lost_datapoints)
            
            file = open('results.txt','a')
            items = [i, laz_file, pct_kept_powerline, pct_lost_datapoints, amount_points, amount_wire, new_amount_points, pct_lost_datapoints, amount_wire-new_amount_wire, self.get_params()]
            for item in items[:-1]:
                file.write(str(item)+",")
            file.write(str(items[-1])+"\n")
            file.close()
        
        alpha = 0.95
        score = alpha * np.mean(pct_kept_powerline_list) + (1-alpha)*np.mean(pct_lost_datapoints_list)
        print("Finished Iter with score: ", score)
        return score
        
def GenPath(path):
        if path[-1] == '/':
            return path
        else:
            return path+'/'

def GetPathRelations(path):
    full_path_to_data = GenPath(path)
    
    ground_removed_image_paths = []
    laz_point_cloud_paths = []
    
    # Find full path to all images
    for path in glob.glob(full_path_to_data+'ImagesGroundRemovedSmall/*'):
        ground_removed_image_paths.append(path)

    # Find full path to all laz files
    for path in glob.glob(full_path_to_data+'LazFilesWithHeightRemoved/*'):
        laz_point_cloud_paths.append(path)
        
    ground_removed_image_paths.sort()
    laz_point_cloud_paths.sort()
    assert len(ground_removed_image_paths)==len(laz_point_cloud_paths), 'Length of images did not match length of laz'

    return ground_removed_image_paths, laz_point_cloud_paths


parser = argparse.ArgumentParser(description='Path to data folder.')
parser.add_argument('folder', type=str, help='folder with data')
args = parser.parse_args()
dir = args.folder

if __name__ == "__main__":

    n_cpu = os.cpu_count()
    print("Number of CPUs in the system:", n_cpu)

    cv = [(slice(None), slice(None))]
    params = {
            "path": Categorical([dir]),
            "canny_lower": Integer(5,40),
            "canny_upper": Integer(50,200),
            "hough_lines_treshold": Integer(5,150),
            "min_line_length": Integer(1,50),
            "max_line_gap": Integer(1,50),
            "closing_kernel_size": [1, 3, 5, 9],
            "opening_kernel_size": [1, 3, 5, 9],
            "small_dialation_kernel": [1, 3, 5, 7],
            "meters_around_line": Integer(3,10),
            "simplify_tolerance": [8],
            "cc_area": Integer(10,10000),
            "distance_resolution": Integer(1,10)
    }

    opt = BayesSearchCV(
        TemplateClassifier(),search_spaces=params,
        cv=cv,
        n_iter=70,
        n_jobs=n_cpu-1,
        random_state=0
    )
    
    # executes bayesian optimization
    X = [[1,2],[1,2],[1,2],[1,2]]
    Y = [1,2,1,2]

    print("Started Fit")
    _ = opt.fit(X,Y)

    # model can be saved, used for predictions or scoring
    print("The best score: ", opt.best_score_)
    print("The best params ", opt.best_params_)
