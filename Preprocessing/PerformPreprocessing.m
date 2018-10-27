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
      images=[images ; dir(fullfile(fullClassDir,'*.jpeg'))];
      images=[images ; dir(fullfile(fullClassDir,'*.JPEG'))];
      % add more image extensions here if necessary

      for kk = 1:length(images)
          if mod(kk,50) == 0
              k/length(dirContents)
              kk/length(images)
          end
          imagePath = images(kk).name;
          fullImagePath = fullfile(fullClassDir, imagePath);
          fullImagePath_out = fullfile(fullClassDir_out, imagePath);
          
          count=count+1;
          
          % If the output file already exists, just continue
          if exist(fullImagePath_out, 'file')
              continue;
          end
          
          % Generate and save the transformed image
          try
            im = imread(fullImagePath);
            % if the image is grayscale make it color
            if size(im,3) ~= 3
               colorim = cat(3,im,im,im); 
               im = colorim;
            end
            im=evalFunc(funcName,im,varargin); 
            imwrite(im, fullImagePath_out);
          catch
            warning('invalid fxn call, moving on...'); 
          end
          
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
    DS_Store="/.DS_Store";
    
   
    last1=string(dir(end-1:end)); 
    last2=string(dir(end-2:end)); 
    last9=string(dir(end-9:end)); 
    

    reject = strcmp(currDir,last1) || strcmp(upDir,last2) || strcmp(DS_Store,last9); 
end

