# GLYPH 

The source code for the paper: "Cross-domain Polygon Extraction from Historical Maps via Legend-guided Semantic Fusion" (Under Review)

This implementation is an instantiation of GLYPH with three expert models. Descriptions of the main parameters are as follows.

```
--img_dir: (str, mandatory) path to the directory storing the source map tif. Naming convention: {map_name}.tif. 
--json_dir: (str, mandatory) path to the directory storing the source map json, pre-parsed to localize the polygon map key in the tif. Naming convention: {map_name}.json.  
--roi_dir: (str, optional) path to the directory storing the region-of-interest binary tif. Naming convention: {map_name}.tif. 

--sol_{a/b/c}: (str, mandatory) path to the directory storing expert output tif. Naming convention: {map_name}_{key_name}.tif. 

--preliminary_dir: (str, optional) path to the directory storing preliminary output files. 
--out_dir: (str, optional) path to the directory storing the output tif. Naming convention: {map_name}_{key_name}.tif. 
```
