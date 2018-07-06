%% Appendix A: Transfer Learning with AlexNet for Object Recognition
% MecE 467 Final Project April 2018
% Mark Sherstan

%% Set Up
% The following program is a modified version of MATLAB's AlexNet Example and
% AlexNet Transfer Learning example as found in references [1] and [2] respectively.
%
% The picture folder (rootFolder) must be in the same directory as the following
% program. Categories are subfolders of the rootFolder with images to be used for
% training and validation purposes. imageDatastore is to be used to effectively
% work with a large collection of images.

clear all
close all

categories = {'fish', 'rex', 'elephant'};

rootFolder = fullfile('images');

imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource', 'foldernames');

%% Sorting Images
% Count number of photos in each label and find the smallest number. Randomly
% minimize all the other categories so that the data is evenly distributed.
% Process the files and then split into 80% for training and 20% for validation.

countCategories = countEachLabel(imds);

minCategoryQty = min(countCategories{:,2});

imds = splitEachLabel(imds, minCategoryQty, 'randomize');

% Process each image to meet net.Layers(1) criteria. See Additional Functions for more details.
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

%% Transfer Learning
% *Extract Training Features*
%
% Load AlexNet and extract training features using the fully connected layer fc7.
% Activations will automatically use a GPU if the Parallel Computing Toolbox and
% a CUDA enabled NVIDIA GPU with compute capability 3.0 or higher [3] is installed.

net = alexnet();

featureLayer = 'fc7';

trainingFeatures = activations(net, trainingSet, featureLayer, 'MiniBatchSize', ...
                               32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;

%%
% *Train Classifier*
%
% Train a multiclass support vector machine (SVM) classifier using a fast linear
% solver.

classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners', ...
            'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% Validate the Model
% Extract features from separate test set and extract the features using AlexNet
% and layer fc7 as before.

testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);

predictedLabels = predict(classifier, testFeatures);

testLabels = testSet.Labels;

%% Output Results
% Tabulate the results using a confusion matrix and output the accuracy and
% confusion matrix plot.

confMat = confusionmat(testLabels, predictedLabels);

confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

accuracy = mean(diag(confMat));

fprintf('The accuracy is %0.1f %%\n',accuracy*100)

% Manipulate data into form MATLAB can plot. See additional functions for
% more details.
confusion_matrix(testLabels,predictedLabels)

%% Additional Functions
% The following two functions are subfunctions used above.
%
%
% <include>readAndPreprocessImage.m</include>
%
%
% <include>confusion_matrix.m</include>

%% References
% The following code was modified and referenced for the development of the above program.
%
% [1] https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html
%
% [2] https://www.mathworks.com/help/nnet/ref/alexnet.html
%
% [3] https://www.mathworks.com/help/nnet/ref/activations.html
%
% [4] https://www.mathworks.com/matlabcentral/fileexchange/57116-deep-learning-for-computer-vision-demo-code?focused=6805482&tab=function
%
% [5] https://www.mathworks.com/matlabcentral/answers/338244-how-can-i-plot-a-confusion-matrix-for-a-multi-class-or-non-binary-classification-problem
