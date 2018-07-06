%% Appendix B: Webcam Object Detection, Tracking and Classification Using a Kalman Filter and Convolutional Neural Network
% MecE 467 Final Project April 2018
% Mark Sherstan - 1392849

%% Track a Single Object Using Kalman Filter and Webcam
% The following program is a heavily modified version of MATLAB's "Using Kalman
% Filter for Object Tracking" example as found in reference [1].
%
% Use deepNetworkAlex to get the network and train the classifier. DeepLearnRexFishElephant.mat
% can be loaded into the MATLAB workspace to save time and computation power.
% The following top level variables are used as global variables.
%
% Input to function is "net" and "classifier" defined by deepNetworkAlex.m or
% DeepLearnRexFishElephant.mat and "n"  is number of video frames to play.

function webcamKalmanFilterNet(net,classifier,n)

frame            = [];  % A video frame
detectedLocation = [];  % The detected location
trackedLocation  = [];  % The tracked location
label            = '';  % Label for the object
utilities        = [];  % Utilities used to process the video
bbox             = [];  % Bounding box location and size

%%
% The following function detects, tracks, and annotates an object when connected
% to a webcam.

function trackSingleObject(param,n)
  % See Defining Default Parameters and Utilities for further information.
  utilities = createUtilities(param);

  isTrackInitialized = false;
  i = 0;

  while i < n % Stop after reading n frames
    frame = readFrame();

    % Detect object location if available
    [detectedLocation, isObjectDetected] = detectObject(frame);

    if ~isTrackInitialized
        if isObjectDetected
          % Initialize a track by creating a Kalman Filter when the object is detected
          % for the first time.
          initialLocation = computeInitialLocation(param, detectedLocation);
          % See getDefaultParameters for specific values
          kalmanFilter = configureKalmanFilter(param.motionModel, ...
                        initialLocation, param.initialEstimateError, ...
                        param.motionNoise, param.measurementNoise);
          isTrackInitialized = true;
          % Used to reduce noise in the tracking system
          trackedLocation = correct(kalmanFilter, detectedLocation);
        else
          trackedLocation = [];
        end
    else
        if isObjectDetected
          % Used to reduce noise in the tracking system
          trackedLocation = correct(kalmanFilter, detectedLocation);
        else
          trackedLocation = [];
        end
    end

    annotateTrackedObject();
    i = i + 1;

  end
end

%%
% Run and visualize program with validated Kalman configuration.

param = getDefaultParameters();
trackSingleObject(param,n);

%% Defining Default Parameters and Utilities
% Get default parameters for creating Kalman Filter and for segmenting the object.
% Values are determined experimentally to work best for the given experimental
% setup. Values are based on MATLAB example [1].

function param = getDefaultParameters
  param.motionModel           = 'ConstantVelocity';
  param.initialLocation       = 'Same as first detection';
  param.initialEstimateError  = 1E5 * ones(1, 2);
  param.motionNoise           = [25, 10];
  param.measurementNoise      = 25;
end

%%
% Create utilities for reading video, detecting moving objects, extracting
% foreground, analyzing connected components, and displaying the results.
% Values are determined experimentally to work best for the given experimental
% setup. Reference [2] was used as an aid in retrieving a video feed and a
% basis for the foreground and blob parameters.

function utilities = createUtilities(param)
  % Acquire input video stream from webcam
  utilities.videoReader = imaq.VideoDevice(                    ...
                                  'winvideo', 1,               ...
                                  'YUY2_640x480',              ...
                                  'ROI', [1 1 640 480],        ...
                                  'ReturnedColorSpace', 'rgb');

  % Remove backlight compensation to reduce major noise in foreground detection
  set(utilities.videoReader.DeviceProperties,'BacklightCompensation','off')

  % Acquire input video properties
  vidInfo = imaqhwinfo(utilities.videoReader);

  % Publish video to video player with given properties
  utilities.videoPlayer = vision.VideoPlayer(                  ...
                                'Name','Final Video',          ...
                                'Position',[100 100 vidInfo.MaxWidth+20 vidInfo.MaxHeight+30]);

  % Set foreground detector parameters based off optimal experimental results
  utilities.foregroundDetector = vision.ForegroundDetector(    ...
                                'NumTrainingFrames', 50,       ...
                                'InitialVariance', (30/255)^2, ...
                                'LearningRate', 0.01,          ...
                                'MinimumBackgroundRatio', 0.7);

  % Set blob analyzer parameters based off optimal experimental results
  utilities.blobAnalyzer = vision.BlobAnalysis(                ...
                                'MaximumCount', 1,             ...
                                'AreaOutputPort', false,       ...
                                'MinimumBlobArea', 2000,       ...
                                'CentroidOutputPort', true,    ...
                                'BoundingBoxOutputPort', true);
end

%% Sub Functions
% The following sub functions are continuously called in accordance to the
% trackSingleObject() major function based off whether an object has been
% initialized or detected.

%%
% Request and read the next video frame from the webcam.

function frame = readFrame()
  frame = step(utilities.videoReader);
end

%%
% Detect objects based on foreground detection and blob analysis.

function [detection, isObjectDetected] = detectObject(frame)
  % convert to gray image for background detection.
  grayImage = rgb2gray(frame);
  utilities.foregroundMask = step(utilities.foregroundDetector, grayImage);
  [detection, bbox] = step(utilities.blobAnalyzer, utilities.foregroundMask);

  if isempty(detection)
    isObjectDetected = false;
  else
    % Must convert frame from single data type to uint8 for CNN to classify
    img = im2uint8(frame);
    img = imcrop(img, bbox);
    label = labelMachine(img,net,classifier);
    isObjectDetected = true;
  end
end

%%
% Find the initial location for tracking purposes and for the Kalman Filter.

function loc = computeInitialLocation(param, detectedLocation)
  if strcmp(param.initialLocation, 'Same as first detection')
    loc = detectedLocation;
  else
    loc = param.initialLocation;
  end
end

%%
% Bound tracked object with bounding box and apply label based off the CNN prediction.

function annotateTrackedObject()
  combinedImage = frame;

  if ~isempty(trackedLocation)
    combinedImage = insertObjectAnnotation(     ...
                    combinedImage, 'rectangle', ...
                    bbox, {label},              ...
                    'Color', 'red');
  end

  step(utilities.videoPlayer, combinedImage);
end

%%
% Use the trained CNN from deepNetworkAlex and predict the object based off the
% cropped image of the bounding box from the foreground detection and blob analysis.

function label = labelMachine(img,net,classifier)
  featureLayer = 'fc7';

  % Pre-process the image as required for the CNN
  img = imresize(img, [227 227]);

  % Extract image features using the CNN
  imageFeatures = activations(net, img, featureLayer);

  % Make a prediction using the classifier
  label = char(predict(classifier, imageFeatures));
end

end

%% References
% The following code was modified and referenced for the development of the above program.
%
% [1] https://www.mathworks.com/help/vision/examples/using-kalman-filter-for-object-tracking.html
%
% [2] https://www.mathworks.com/matlabcentral/fileexchange/40195-how-to-detect-and-track-white-colored-object-in-live-video
