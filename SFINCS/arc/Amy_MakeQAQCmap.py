#uses one of 4 regional templates of RP000 scenario for entire SLR suite (0-3m)
#and replaces the source data with shapefiles for a specified storm return period (RP000, RP001, RP020, or RP100)
#and saves output to a new mxd for QA/QC analyses

#***after running, you must open the new output MXD and manually update the RP value listed in the layer names***********
# listed in the Table of Contents e.g., change SC_ponding_SLR000_RP000 to SC_ponding_SLR000_RP001 to match new data source
#*************************************************************************************************************************
import arcpy
from arcpy import env
# Reset geoprocessing environment settings to clear extent and mask setting from other scripts
arcpy.ResetEnvironments()
env.overwriteOutput = "True"

#-------------------CHANGE THESE INPUT VARIABLES FOR EVERY RUN-----------------------------------
#select regions included in the input map template
region_list = ["GA"]   ### select from ["VA","NC"] ["SC","GA"] OR ["FL"]

#select corresponding file naming convention for above regions
reg_names = "GA"   #### "VA_NC" "SC_GA" OR "FL"

#select a single storm RP you want to use in the new output map 
outputRP = "RP000"   ### "RP000" "RP001" "RP020" OR "RP100"
#------------------------------------------------------------------------------------------------

root_drive = "F:/"
root_dir = root_drive + "FloSup/FloSup_vMay17_2022_v3/"  #directory containing data you want in output map

#this script uses regional map templates containing all SLR shapefiles for a single storm RP
QAQC_WS = root_drive + "FloSup/QAQC_maps_4_review/"        #output workspace for new maps
template_WS = root_drive + "FloSup/QAQCmap_templates/"     #location  of existing map templates
maptemplate = template_WS + reg_names + "_allSLRs_1storm_template_4_DataReview.mxd"    #existing template
templateRP = "RP000"                                     #the RP scenario used in the above map template
print "input template is " + maptemplate

new_mxd = QAQC_WS + reg_names + "_" + outputRP + "_allSLRs_June2022_v3.mxd"  ###append current date to output mxd filename
print "output map is " + new_mxd

#set up to run through entire SLR suite for a single storm RP
SLR_list = ["000", "025", "050", "100", "150", "200", "300"]

#------------------------------------------------------------------------------------------------
in_mxd = arcpy.mapping.MapDocument(maptemplate)
lyrlist = arcpy.mapping.ListLayers(in_mxd)

for reg in region_list:
	print "QA map group is " + reg
	for SLRnum in SLR_list:
		SLR_var = "SLR" + str(SLRnum)
		print "SLR is " + SLR_var

		for lyr in lyrlist:
			#workspace containing the new shapfiles to be used
			newdata_WS = root_dir + outputRP + "/" + SLR_var + "_" + outputRP + "/display_shps/"
			#print "getting data from " + newdata_WS

			#flood hazard layer name in template Table of Contents
			infldlyr_name = reg + "_flood_hazard_" + SLR_var + "_" + templateRP
			#print "in flood layer to update is " + infldlyr_name
			newfldshp = reg + "_flood_hazard_" + SLR_var + "_" + outputRP
			#print "replacing fld polys with " + newdata_WS + newfldshp

			#ponding layer name in template Table of Contents
			inpondlyr_name = reg + "_ponding_" + SLR_var + "_" + templateRP
			#print "in pond layer to update is " + inpondlyr_name
			newpondshp = reg + "_ponding_" + SLR_var + "_" + outputRP
			#print "replacing ponds with " + newdata_WS + newpondshp

			if lyr.name == infldlyr_name:
				print "flood layer is " + lyr.name
				print "replace WS is " + newdata_WS
				lyr.replaceDataSource(newdata_WS, "SHAPEFILE_WORKSPACE", newfldshp, True)

			if lyr.name == inpondlyr_name:
				print "pond layer is " + lyr.name
				print "replace WS is " + newdata_WS
				lyr.replaceDataSource(newdata_WS, "SHAPEFILE_WORKSPACE", newpondshp, True)
in_mxd.saveACopy(new_mxd)
del in_mxd 
print "now open " + new_mxd + " and manually update RP value in TOC to match the new data source"
