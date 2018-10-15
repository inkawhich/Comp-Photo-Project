15 Oct 2018
Benjamin Flowers
PerformPreprocessing.m & PreprocessScript.m Usage

 ------------------------------------------------------ 
| 			Purpose:          	       |
 ------------------------------------------------------
	
This function loops through a directory of classes, each containing some set of images, and  processes the images, storing the results in a directory structure mirroring the original. 

 ------------------------------------------------------ 
| 		Image Directory structure:             |
 ------------------------------------------------------

Images (point here)
	Dogs	
		im0.jpg
		im1.jpg
	Cats
		im0.jpg
		im1.jpg
		im2.jpg
	Birds
		im0.jpg
Images_Transform (created by script)
	Dogs	
		im0.jpg
		im1.jpg
	Cats
		im0.jpg
		im1.jpg
		im2.jpg
	Birds
		im0.jpg

Pass to the function the name of the directory containing the classes you intend to process. It should contain a number of directories each containing classes, which themselves contain images. The function will check to see if the corresponding output directories exist, and will create them if necessary and populate them. The overall output directory name is the original appended by the name of the function supplied. See the section on Function Arguments / Usage for details. 

 ------------------------------------------------------ 
| 		    Function Location:                 |
 ------------------------------------------------------

It might be necessary to place PerformPreprocessing.m in the directory where the image processing functions you need are located. For example, I placed it in the BIMEF directory to use the functions contained there. If you are using generic Matlab functions the location doesn't matter. 

 ------------------------------------------------------ 
| 		Function Arguments / Usage             |
 ------------------------------------------------------

PerformPreprocessing.m accepts a minimum of two arguments: inputDirectory is a string indicating the directory that contains the image classes you want to process, as discussed above. The second argument, funcName, is a string corresponding to the function you want to call on the input images. For example, "rgb2gray". After the first 2 arguments, you can supply additional arguments that will be passed as args to the function you provide. For example, you could call PerformPreprocessing("C:/ImageData","imrotate",180) to rotate each image 180 degrees. These additional arguments will appear in the name of the output directory. Specifically "C:/ImageData_imrotate_180/" will be created and populated.


 -------------------------------------------------------
| 		    PreprocessScript.m:                 |
 -------------------------------------------------------

The script PreprocessScript.m shows several example calls to PerformPreprocessing.m. Note that various numbers of arguments are supplied and that generic and specific image processing functions are called. BIMEF source: https://github.com/baidut/BIMEF. 
 
 ------------------------------------------------------ 
| 		    Future Improvements                |
 ------------------------------------------------------

The following may need to be improved or added:
-Parallelization
-Additional image file-type parsing/support (.bmp, .tif, etc)
-Extra Path handling for image functions? 