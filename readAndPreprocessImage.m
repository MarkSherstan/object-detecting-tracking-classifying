function Iout = readAndPreprocessImage(filename)
% Resize the image as required for the CNN. See reference [4].

img = imread(filename);

Iout = imresize(img, [227 227]);

end
