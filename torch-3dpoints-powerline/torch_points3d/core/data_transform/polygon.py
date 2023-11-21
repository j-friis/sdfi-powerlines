import shapely
from shapely.geometry import Polygon, mapping
import numpy as np
import cv2
import laspy
import rasterio
from rasterio.features import shapes
from matplotlib.path import Path as plt_path
import ipdb

class HoughLinePre(object):

    def __init__(self, path_to_data, canny_lower=19, canny_upper=101, hough_lines_treshold=30, max_line_gap=6, min_line_length=12, meters_around_line=10, cc_area=1500, simplify_tolerance=8, small_dialation_kernel=5):
        self.path_to_data = path_to_data
        #self.filename = filename
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper
        self.hough_lines_treshold = hough_lines_treshold
        self.max_line_gap = max_line_gap
        self.min_line_length = min_line_length
        self.meters_around_line = meters_around_line
        self.cc_area = cc_area
        self.move_in = 0.01
        self.simplify_tolerance = simplify_tolerance
        self.small_dialation_kernel = small_dialation_kernel

    def __call__(self, filename):
        lines_image = self.ImageProcessing(filename)
        reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons = self.Polygonize(lines_image)
        point_cloud = laspy.read(self.path_to_data+'/LazFilesWithHeightParam/'+filename+'_hag_nn.laz', laz_backend=laspy.compression.LazBackend.LazrsParallel)
        indexes_needed = self.FilterPolygons(reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, lines_image)
        new_las = self.Predictions(indexes_needed, point_cloud)
        return new_las
    
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

    def ImageProcessing(self, filename):
        # Load Image
        image = cv2.imread(self.path_to_data+'/ImagesGroundRemoved/'+filename+'_max.tif', cv2.IMREAD_UNCHANGED)
        image = np.where(image >= 0, image, 0)
        image = image/np.max(image)
        image = (image*255).astype(np.uint8)

        # Apply canny edge detection
        image = cv2.Canny(image, self.canny_lower, self.canny_upper, None, 3)

        # Draw the lines from The Probabilistic Hough Line Transform
        lines_image = np.zeros_like(image)
        linesP = cv2.HoughLinesP(
            image, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=self.hough_lines_treshold, # Min number of votes for valid line
            minLineLength=self.min_line_length, # Min allowed length of line
            maxLineGap=self.max_line_gap # Max allowed gap between line for joining them
            )
        
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
            if not self.BBTouchingEdge(image.shape, bounding_box[i], 10):
                area = bounding_box[i][cv2.CC_STAT_AREA]
                if area < self.cc_area:
                    lines_image[label_ids == i] = 0

        # Get pixels per meter to create a cirkular kernel size of size "meters_around_line"
        x_pixels, y_pixels = image.shape
        x_pixels, y_pixels = x_pixels/1000, y_pixels/1000

        meters_around_line = self.meters_around_line
        kernel_size = int(meters_around_line*np.ceil(x_pixels))
        circular_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Create a cirkular kernel using (image, center_coordinates, radius, color, thickness)
        cv2.circle(circular_kernel, (int(kernel_size/2), int(kernel_size/2)), int(kernel_size/2), 255, -1)
        # Apply dilation using the cirkular kernel
        lines_image = cv2.dilate(lines_image, circular_kernel, iterations=1)
        return lines_image    

    def Polygonize(self, lines_image):
        # Create Polygons and Multi Polygons
        mask = (lines_image == 255)
        output = rasterio.features.shapes(lines_image, mask=mask, connectivity=4)
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
            bbox_all_polygon_path.append(plt_path(bb))

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
                tmp_multi_pol_boxes.append(plt_path(bb))
            bbox_all_multi_polygons_path.append(tmp_multi_pol_boxes)

        # Create Path polygons from the simplified shapely polygons
        simplified_all_polygons_path = [plt_path(mapping(p)['coordinates'][0]) for p in simplified_all_polygons]
        simplified_all_multi_polygons_path = []
        for multi_pol in simplified_all_multi_polygons:
            tmp = [plt_path(mapping(p)['coordinates'][0]) for p in multi_pol]
            simplified_all_multi_polygons_path.append(tmp)

        return simplified_all_polygons_path, simplified_all_multi_polygons_path, bbox_all_polygon_path, bbox_all_multi_polygons_path
    

    def MaxMinNormalize(self, arr):
        return (arr - np.min(arr))/(np.max(arr)-np.min(arr))

    def CastAllXValuesToImage(self, arr, x_pixels):
        return self.MaxMinNormalize(arr)*x_pixels

    def CastAllYValuesToImage(self, arr, y_pixels):
        return (1-self.MaxMinNormalize(arr))*y_pixels

    def FilterPolygons(self, reg_polygons, multi_polygons, bbox_reg_polygon, bbox_multi_polygons, point_cloud, image):
        # Pixels per kilometer
        x_pixels, y_pixels = image.shape

        x_values = self.CastAllXValuesToImage(point_cloud.X, x_pixels)
        y_values = self.CastAllYValuesToImage(point_cloud.Y, y_pixels)

        # As there are some problems with pixels around the edges after we have simplified
        # we need to move the some of the points into the image again 
        x_values = np.where(x_values <= (x_pixels-1)-self.move_in, x_values, (x_pixels-1)-self.move_in)
        x_values = np.where(x_values >= self.move_in, x_values, self.move_in)
        
        y_values = np.where(y_values <= (y_pixels-1)-self.move_in, y_values, (y_pixels-1)-self.move_in)
        y_values = np.where(y_values >= self.move_in, y_values, self.move_in)


        # Format: [(1,1), (3,5), (1,5), ...] with 30 mio samples
        list_zipped = np.array(list(zip(x_values, y_values)))

        # Generate a bool list to obtain the final indexes from the dataset
        indexes_needed = np.zeros(len(x_values), dtype=bool)

        # Run through all polygons and check which points are inside the polygon
        for i in range(len(reg_polygons)):
            # Check if point is inside the bounding box
            indexes_inside_box = bbox_reg_polygon[i].contains_points(list_zipped)
            indexes_inside_box = np.array([index for index, x in enumerate(indexes_inside_box) if x])
            if len(indexes_inside_box) != 0:
                # Generate small dataset
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

                if len(indexes_inside_box) != 0:
                    # Generate small dataset
                    tmp = list_zipped[indexes_inside_box]

                    # Check if any of these points are in the polygon
                    indexes_inside_polygon = simpli_multi_pol[j].contains_points(tmp)
                    final_indexes = indexes_inside_box[indexes_inside_polygon]

                    # Update the indexes
                    tmp_indexes_not_needed[final_indexes] = 1
                indexes_needed = indexes_needed | (tmp_indexes_needed & np.invert(tmp_indexes_not_needed))
                    
        return indexes_needed
    
    def Predictions(self, indexes_needed, point_cloud):
        new_point_cloud = point_cloud[indexes_needed]
        return new_point_cloud
