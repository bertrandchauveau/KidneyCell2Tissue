#to convert a geojson file encoding whole cell annotations into a ensemble of tiles as a cell masks (one color per category, see below)
import os
import shutil
import numpy as np
import natsort
from natsort import os_sorted
import geojson
from shapely.geometry import shape, Polygon, MultiPolygon, Point, box
from shapely.strtree import STRtree
import cv2
import rasterio
from rasterio.features import rasterize
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert GeoJSON annotations to a whole cell mask.")
    
    # Define the arguments
    parser.add_argument('--wsi_name', type=str, default='test.tiff', help='Name of the WSI file')
    parser.add_argument('--geojson_path', type=str, required=True, help='Path to the folder containing geojson')
    parser.add_argument('--geojson_name', type=str, required=True, help='Name of the geojson file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--wsi_height', type=int, default=10000, help='Height of WSI')
    parser.add_argument('--wsi_width', type=int, default=10000, help='Width of WSI')
    parser.add_argument('--tile_size', type=int, default=1024, help='Size of tile (square)')
    parser.add_argument('--overlap', type=float, default=0.125, help='Overlap percentage')
    
    args = parser.parse_args()

    #placeholder
    tile_size = (args.tile_size, args.tile_size)
    overlap = args.overlap
    WSI_height, WSI_width = args.wsi_height, args.wsi_width
    WSI_name = args.wsi_name
    semantic = True # Kept as True based on your original code
    path_output = args.output_dir
    path_to_geojson = args.geojson_path
    geojson_name = args.geojson_name
    
    #list of cell classes (modify if necessary)
    cell_class = ['PT', 'PT_injured', 'DCT', 'CD_principal', 'Intercalated_cells', 'TAL', 'Endothelial', 
                'Lymphatic_endothelial', 'Pericytes_vSMC', 'Fibroblasts', 'PEC', 'Podocytes', 
                'B_cells', 'CD4_T', 'CD8_T', 'Dendritic_cells', 'Macrophages', 'Mast_cells', 'NK', 'Plasma_cells', 'T_γδ']
    
    #dictionary mapping a cell type to a specific color
    #here, cells were merged into categories that share the same color
    #inflammatory, tubule, parietal epithelial cell, podocyte, endothelial, smooth muscle
    cell_mapping_semantic = {'B_cells': [227,73,170],
             'CD4_T': [227,73,170],
             'CD8_T': [227,73,170],
             'CD_principal': [135,106,78],
             'DCT': [135,106,78],
             'Dendritic_cells': [227,73,170],
             'Endothelial': [135,131,56],
             'Fibroblasts': [135,57,57],
             'Intercalated_cells': [135,106,78],
             'Lymphatic_endothelial': [135,131,56],
             'Macrophages': [227,73,170],
             'Mast_cells': [227,73,170],
             'NK': [227,73,170],
             'PEC': [73,129,43],
             'Pericytes_vSMC': [3, 125, 133],
             'Plasma_cells': [227,73,170],
             'Podocytes': [106,104,136],
             'PT': [135,106,78],
             'PT_injured': [135,106,78],
             'T_γδ': [227,73,170],
             'TAL': [135,106,78],
             'background': [0, 0, 0]}
    
    def load_geojson_with_names(file_path):
        with open(file_path, 'r') as file:
            data = geojson.load(file)
    
        polygons_with_names = []
        for feature in data['features']:
            geom = shape(feature['geometry'])
            name = feature['properties'].get('classification', {}).get('name', None)
                
            if isinstance(geom, Polygon):
                polygons_with_names.append((geom, name))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    polygons_with_names.append((poly, name))
        
        return polygons_with_names
    
    def get_tile_mask(polygons_with_names, tile_box, tile_shape, class_OI, tile_origin_x, tile_origin_y):
        """
        Args:
            polygons_with_names: list of (Polygon, name)
            tile_box: The Shapely box representing the tile boundaries
            tile_shape: (height, width) of the tile
            class_OI: List of classes to include
            tile_origin_x: The x-coordinate of the top-left of the tile
            tile_origin_y: The y-coordinate of the top-left of the tile
        """
        mask = np.zeros(tile_shape, dtype=np.uint16)
        instance_dict = {}
        
        # Filter and CLIP polygons to the tile boundary
        # This is crucial: we clip them to the box, then shift them to (0,0)
        relevant_shapes = []
        for polygon, name in polygons_with_names:
            if polygon.intersects(tile_box):
                # 1. Clip the geometry so it doesn't extend beyond tile boundaries
                clipped_poly = polygon.intersection(tile_box)
                
                # 2. IMPORTANT: Shift the geometry coordinates!
                # We translate the clipped polygon from Global Space to Tile Local Space
                # (New_x = Old_x - Origin_x)
                from shapely.affinity import translate
                local_poly = translate(clipped_poly, xoff=-tile_origin_x, yoff=-tile_origin_y)
                
                relevant_shapes.append((local_poly, name))
    
        if not relevant_shapes:
            return mask, instance_dict
    
        i = 1
        for polygon, class_name in relevant_shapes:
            if class_name in class_OI:
                # Prepare shapes for rasterize (handle MultiPolygon)
                shapes_to_rasterize = []
                if isinstance(polygon, MultiPolygon):
                    for part in polygon.geoms:
                        shapes_to_rasterize.append((part, i))
                else:
                    shapes_to_rasterize.append((polygon, i))
                
                # Rasterize onto the local tile canvas
                mask_obj = rasterize(
                    shapes_to_rasterize, 
                    out_shape=tile_shape, 
                    fill=0, 
                    dtype=np.uint16
                )
                
                # Update mask: if pixels were already set by a previous polygon, 
                # we keep the new ID (or you can implement logic to prevent overlap)
                mask = np.where(mask_obj != 0, mask_obj, mask)
                
                instance_dict[i] = class_name
                i += 1
                
        return mask, instance_dict
    
    
    #load geojson file
    polygons_with_names = load_geojson_with_names(os.path.join(path_to_geojson, geojson_name))
    
    
    # Build the STRtree for fast spatial querying
    geometries = [item[0] for item in polygons_with_names]
    tree = STRtree(geometries)
    
    height, width = tile_size
    overlap_h, overlap_w = int(height * overlap), int(width * overlap)
    
    for y in range(0, WSI_height + 1, height - overlap_h):
        if y + height > WSI_height: y = WSI_height - height
        for x in range(0, WSI_width + 1, width - overlap_w):
            if x + width > WSI_width: x = WSI_width - width
    
            # 1. Define tile bounding box
            tile_box = box(x, y, x + width, y + height)
            
            # 2. STRtree Query: Find indices of polygons whose bounding box intersects tile_box
            indices = tree.query(tile_box)
            
            # Check if any returned polygon actually intersects the tile (not just its bounds)
            intersecting_polys_for_this_tile = []
            for idx in indices:
                poly = geometries[idx]
                if poly.intersects(tile_box):
                    intersecting_polys_for_this_tile.append(polygons_with_names[idx])
            
            if intersecting_polys_for_this_tile:
                # 3. Only compute mask for this specific tile
            
                tile_mask, instance_dict = get_tile_mask(
                    intersecting_polys_for_this_tile, 
                    tile_box, 
                    (height, width), 
                    cell_class,
                    tile_origin_x=x, 
                    tile_origin_y=y
                )
                # 4. Convert to semantic if requested => here a color mask
                if semantic:
                    semantic_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    for inst_id, class_name in instance_dict.items():
                        val = cell_mapping_semantic[class_name]
                        semantic_mask[tile_mask == inst_id] = val
                    tile_mask = semantic_mask
            
                # 5. Save
                cv2.imwrite(f'{os.path.join(path_output, WSI_name[:-5])}_[x={x},y={y},w={width},h={height}].png', tile_mask[...,::-1])
            
if __name__ == "__main__":
    main()
