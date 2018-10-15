

% Example 1 
inputDirectory='/Users/benjamincflowers/Desktop/MyData';
funcName="rgb2gray"; 
PerformPreprocessing(inputDirectory,funcName); 

% Example 2 
inputDirectory='/Users/benjamincflowers/Desktop/MyData';
funcName="imrotate"; 
PerformPreprocessing(inputDirectory,funcName,42); 

% Example 3 
inputDirectory='/Users/benjamincflowers/Desktop/MyData';
funcName="imrotate"; 
PerformPreprocessing(inputDirectory,funcName,142,'bilinear','loose'); 

% Example 4 
inputDirectory='/Users/benjamincflowers/Desktop/MyData';
funcName="BIMEF"; 
PerformPreprocessing(inputDirectory,funcName); 



