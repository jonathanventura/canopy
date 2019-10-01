import fiona
from pyproj import Proj, Transformer, transform
import rasterio
from rasterio import features

def load_and_transform_shapefile(uri,property_name,target_crs=None):
  """ Loads polygons and their corresponding labels from a shapefile
      and optionally transforms its coordinates to a desired
      coordinate reference system (CRS).
      Arguments:
        uri: URI of shapefile (could be local path or Amazon S3 URI)
        property_name: the name of the property containing the class label
        target_crs: CRS to which coordinates will be transformed;
                    defaults to None, in which case no transformation is performed.
      Returns:
        List of polygons (each polygon is a list of coordinates)
        List of labels (integers)
  """

  # open the shapefile
  with fiona.open(uri, "r") as shapefile:
    
    # extract labels
    labels = [feature['properties'][property_name] for feature in shapefile]

    # extract polygons
    polygons = [feature['geometry'] for feature in shapefile]

    # transform coordinates if necessary
    if target_crs is not None:

      # get the CRS of the polygons
      src_crs = shapefile.crs['init']

      # make a proj object for the source CRS
      src_proj = Proj(init=src_crs)

      # make a proj object for the target CRS
      target_proj = Proj(init=target_crs)

      # make transformer object for fast transformations
      transformer = Transformer.from_proj(src_proj,target_proj)

      # transform coordinates of each polygon
      for poly in polygons:
        for i,coords in enumerate(poly['coordinates'][0]):
          # transform coordinates and save back into array
          x,y = transformer.transform(coords[0], coords[1])
          poly['coordinates'][0][i] = (x,y)

  return polygons, labels

def rasterize_shapefile(polygons, labels, metadata, uri):
  """ Rasterizes polygons to a file.
      Arguments:
        polygons: list of polygons
        labels: list of labels (integers)
        metadata: metadata dictionary containing target CRS
        uri: URI for output file (can be local file or S3 URI)
      Returns:
        Numpy array of the raster
  """

  # copy the metadata and set the data type to uint8, no-data-value to 255
  labels_meta = metadata.copy()
  labels_meta['dtype'] = 'uint8'
  labels_meta['nodata'] = 255
  labels_meta['count'] = 1

  # open the raster file to be written
  with rasterio.open(uri, 'w', compress='lzw', **labels_meta,) as out:
    # get all polygons and labels
    shapes = ((geom,value) for geom, value in zip(polygons, labels))

    # create the raster
    raster = features.rasterize(shapes=shapes, fill=labels_meta['nodata'], out_shape=out.shape, transform=out.transform)

    # write the raster out to file
    out.write_band(1, raster)

  return raster
