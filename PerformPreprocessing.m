function f = PerformPreprocessing(inputDirectory,funcName,varargin  )

    close all;
    
    outSuffix=""; 
    for v=varargin
        outSuffix = strcat(outSuffix,"_",string(v));
    end
   
    count=0; 
    
    outDirectory = strcat(string(inputDirectory),"_",funcName,outSuffix);
    

    if (exist(outDirectory,'dir') ~= 7)
        mkdir(char(outDirectory)); 
    end
    
    dirContents=dir(inputDirectory);

    for k = 1:length(dirContents)
      classDir = dirContents(k).name;
      fullClassDir = fullfile(inputDirectory, classDir);
      
      if (rejectDir(fullClassDir))
          continue;
      end; 
     
      fullClassDir_out = fullfile(char(outDirectory), classDir);
      if (exist(fullClassDir_out,'dir') ~= 7)
        mkdir(char(fullClassDir_out)); 
      end

      images= dir(fullfile(fullClassDir,'*.jpg'));
      images=[images ; dir(fullfile(fullClassDir,'*.png'))];
      % add more image extensions here if necessary

      for kk = 1:length(images)
          imagePath = images(kk).name;
          fullImagePath = fullfile(fullClassDir, imagePath);
          fullImagePath_out = fullfile(fullClassDir_out, imagePath);
          
          count=count+1;
          
          im = imread(fullImagePath);
          im=evalFunc(funcName,im,varargin); 
          imwrite(im, fullImagePath_out);
          
      end

    end
    
    disp(strcat("Processed ", num2str(count) , " images. Destination directory: ", string(outDirectory))); 

end

% evaluate function with args 
function im = evalFunc(funcName,im,varargin) 
    
    args=varargin{1}; 
    L = length(args);
    
    if (L==0)
        im=feval(funcName,im);
    elseif (L==1)
        im=feval(funcName,im,args{1});
    elseif (L==2)
        im=feval(funcName,im,args{1},args{2});
    elseif (L==3)
        im=feval(funcName,im,args{1},args{2},args{3});
    elseif (L==4)
        im=feval(funcName,im,args{1},args{2},args{3},args{4});
    end
    
end

% reject . , .. , & .DS_Store directories
function reject = rejectDir(dir)


    currDir="/.";
    upDir="/..";
    DS_Store="/.DS_Store"
    
   
    last1=string(dir(end-1:end)); 
    last2=string(dir(end-2:end)); 
    last9=string(dir(end-9:end)); 
    

    reject = strcmp(currDir,last1) || strcmp(upDir,last2) || strcmp(DS_Store,last9); 
end
