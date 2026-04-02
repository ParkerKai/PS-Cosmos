The above scripts are used to burn a stream into a hydroflattened dem.


Anaconda Steps:

Step1: Create the geoprocessing environment
	From the anaconda prompt do:
	conda env create -f environment.yml
	
Step2: Activate the Environment
	conda activate geoprocessing

Step3: Run the python scripts
	Fire up spyder (or whatever your favorite ide is).
	Or just run the python scripts from the anaconda prompt
		
######################################################################################

Step1: Digitize a thalweg for the stream channel. 
	Beginning of the thalweg should be the beginning of the hydroflattened dem and end is where you want the burning to end.
	Should be in a projected coordinate sytem. If not you could add some code to project.
	
Step2: Generate transects
	Currently using Arcgis "Generate Transects Along Lines".  
	Distance Between Transects = 50 (25 better resolves the stream but more manual editinG where things don't work
	Transect Length = 300  (just make sure it captures the full stream width otherwise won't be able to pull edges)
	Generate transects at line start and end = unchecked (added transects are out of order with other transects so mess up looping).

Step3: Run "Pull_ChannelEdge.py"
	This scripts goes through and finds the edge of the channel along the transects.  
	Also fixes transects so direction is always the same across the thalweg.  
	
Step4: Open up Edges and make sure the algorithm has pulled the edges of the channel correctly. Edit if not.

Step5: Run "Interpolate_Transects.py"
	If breaks with "Fix Geometry before Proceeding" error, then fix geometry

Step6: import {}_EditPolygon.shp and NewBathy_{}.shp shapefiles into arcgis pro

Step7: Splines with barriers with NewBathy_{}.shp as elevation source and {}_EditPolygon.shp as boundary
	Cell size = 1
	Smoothing Factor = 0.3

Step8: Extract by mask on created raster, with mask being {}_EditPolygon.shp

