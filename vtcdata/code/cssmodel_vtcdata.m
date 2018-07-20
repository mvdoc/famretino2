function cssmodel_vtcdata(subject, roi, debug)
% CSSMODEL_VTCDATA(subject, roi, debug)
%
% This function is a modified version of the cssmodel_example.m script
% available from http://kendrickkay.net/socmodel to be run on the VTCdata
% available from http://kendrickkay.net/vtcdata
% There are three subjects and 14 rois.
%
% Example: cssmodel_vtcdata(1, 1)
if nargin == 2
    debug = 0;
end
% set up number of workers for pool
nworkers = 16;
%nworkers = 3;

%% Add code to the MATLAB path
addpath(genpath(fullfile(pwd,'knkutils')));

%% Load data

dataset = sprintf('../dataset%02d.mat', subject);
load(dataset,'betamn','betase','roilabels');

% Check if output exists already
outdir = '../output';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

fnout = [outdir, '/', sprintf('sub-%02d_', subject), roilabels{roi}];
if debug
    fnout = [fnout '_dbg'];
end

fnout = [fnout '.mat'];

if exist(fnout, 'file')
  error('Output %s exists, not overwriting', fnout);
end

fprintf('#### Running for subject %d, roi number %d (%s)\n', subject, roi, roilabels{roi});

%% Load stimuli

load('../conimages.mat','conimages');

%% Perform stimulus pre-processing

% extract the stimuli we need
stimulus = conimages{1};

% resize the stimuli to 100 x 100 (to reduce computational time)
temp = zeros(100,100,size(stimulus,3));
for p=1:size(stimulus,3)
  temp(:,:,p) = imresize(stimulus(:,:,p),[100 100],'cubic');
end
stimulus = temp;

% ensure that all values are between 0 and 1
stimulus(stimulus < 0) = 0;
stimulus(stimulus > 1) = 1;

% inspect one of the stimuli
% figure;
% imagesc(stimulus(:,:,10));
% axis image tight;
% colormap(gray);
% colorbar;
% title('Stimulus');
%%

% reshape stimuli into a "flattened" format: 69 stimuli x 100*100 positions
stimulus = reshape(stimulus,100*100,196)';

%% Select only the stimuli corresponding to faces
% From http://kendrickkay.net/vtcdata/ :
% "For the pRF-estimation experiment, there are 196 beta weights ordered in 
% four groups of 7*7=49, where the first group consists of the phase-scrambled 
% faces, the second group consists of the small faces, the third group consists 
% of the medium faces, and the fourth group consists of the large faces. 
% In each group, the order is left to right and then top to bottom."
face_idx = 50:196;
stimulus = stimulus(face_idx, :);

%% Prepare for model fitting

% to perform model fitting, we will be using fitnonlinearmodel.m.  this function
% is essentially a wrapper around MATLAB's lsqcurvefit.m function.  the benefit
% of fitnonlinearmodel.m is that it simplifies input and output issues, deals with
% resampling (cross-validation and bootstrapping), makes it easy to evaluate multiple
% initial seeds, and makes it easy to perform stepwise fitting of models.
%
% to prepare for the call to fitnonlinearmodel.m, we have to define various 
% input parameters.  this is what we will now do.

% define constants
res = 100;  % resolution of the pre-processed stimuli

% the parameters of the CSS model are [R C S G N] where
%   R is the row index of the center of the 2D Gaussian
%   C is the column index of the center of the 2D Gaussian
%   S is the standard deviation of the 2D Gaussian
%   G is a gain parameter
%   N is the exponent of the power-law nonlinearity

% define the initial seed for the model parameters
seed = [(1+res)/2 (1+res)/2 res 1 0.5];

% define bounds for the model parameters
bounds = [1-res+1 1-res+1 0   -Inf 0;
          2*res-1 2*res-1 Inf  Inf Inf];

% fitnonlinearmodel.m provides the capacity to perform stepwise fitting.
% here, we define a version of bounds where we insert a NaN in the first
% row in the spot that corresponds to the exponent parameter.  this 
% indicates to fix the exponent parameter and not optimize it.
boundsFIX = bounds;
boundsFIX(1,5) = NaN;

% issue a dummy call to makegaussian2d.m to pre-compute xx and yy.
% these variables are re-used to achieve faster computation.
[d,xx,yy] = makegaussian2d(res,2,2,2,2);

% we will now define a function that implements the CSS model.  this function
% accepts a set of parameters (pp, a vector of size 1 x 5) and a set of stimuli 
% (dd, a matrix of size A x 100*100) and outputs the predicted response to those 
% stimuli (as a vector of size A x 1).  for compactness, we implement this 
% function as an anonymous function where the parameters are given by pp
% and the stimuli are given by dd.
modelfun = @(pp,dd) pp(4)*((dd*vflatten(makegaussian2d(res,pp(1),pp(2),pp(3),pp(3),xx,yy,0,0)/(2*pi*pp(3)^2))).^pp(5));

% notice that the overall structure of the model is 
%   RESP = GAIN*(STIM*GAU).^N
% where STIM*GAU represents the dot product between the stimulus and the 2D Gaussian.
% also, note that the division by (2*pi*pp(3)^2) makes it such that the integral
% of the Gaussian is equal to 1 (this aids the interpretation of model parameters).

% now that we have defined modelfun, we are ready to define the final model
% specification.  in the following, we specify a stepwise fitting scheme.
% in the first fit (the first row), we start at the seed and optimize all 
% parameters except the exponent parameter.  in the second fit (the second row),
% we start at the parameters estimated in the first fit and optimize all parameters.
% the purpose of the stepwise fitting is to help converge to a good solution 
% (i.e. avoid local minima).  (the anonymous functions in the second row accept
% a single input, ss, which refers to the parameters estimated in the first fit.)
model = {{seed       boundsFIX   modelfun} ...
         {@(ss) ss   bounds      @(ss) modelfun}};

% define the resampling scheme to use.  here, we use 0, which
% means to just fit the data (no cross-validation nor bootstrapping).
% resampling = 0;

% define the metric that we will use to quantify goodness-of-fit.
% here, we use a version of the coefficient of determination (R^2)
% in which variance in the data is computed relative to 0.
% this is sensible since the data being fitted are beta weights that
% represent evoked BOLD responses relative to the baseline signal level
% (which corresponds to 0).
metric = @(a,b) calccod(a,b,[],[],0);


%% Data subselection and processing
% take the data corresponding to one roi
data = betamn(:, 1);  % take first experiment only: pRF estimation
% select the ROI
data = data{roi};
% select only the full face stimuli
data = data(:, face_idx);

% filter out nan voxels if it happens
ok_voxs = sum(isnan(data), 2) == 0;
fprintf('Removing %d/%d voxels because contain nans\n', sum(~ok_voxs), length(ok_voxs))
data = data(ok_voxs, :);

% Rectify the data, from supplementary materials
% "For the purposes of modeling, we rectified the response amplitudes observed 
% at each voxel (negative response amplitudes were set to zero). In our 
% experiments, negative BOLD responses can be found in V1?V3, indicating 
% that the presentation of a face can cause the BOLD signal in a voxel to 
% drop below baseline. Such suppression may reflect early attentional 
% filtering and may have a distinct physiological source [S19] compared to 
% positive BOLD responses which are the focus of the present study.
data(data < 0) = 0;

% Define resampling scheme as paper
% from supplementary materials
% "Cross-validation was used to estimate the accuracy of the CSS model. 
% For the Full and Individual models, the response amplitudes for each face 
% size were randomly split into ten groups, and each group was systematically 
% left out and used as the testing set (thus, a total of 30 cross-validation 
% iterations were performed)"

% make a matrix for resampling, each row is a fold, 
% and -1 is testing, 1 is training

% randomly split each face size into 10 groups
resampling = ones(30, 147);
counter = 1;
% set random seed for reproducibility and to use same faces across ROIs
rng(42);
for iface = 1:3
   randidx = randperm(49) + (iface-1)*49;
   step = 1:5:50;
   ig = 1;
   while ig <= 10
       start = step(ig);
       if ig == 10
           finish = 49;
       else
           finish = step(ig + 1) - 1;
       end
       idx = randidx(start:finish);
       resampling(counter, idx) = -1;
       counter = counter + 1;
       ig = ig + 1;
   end
end

% check we have 5 or 4 elements in test set
assert(isequal(unique(sum(resampling == -1, 2)), [4, 5]'))

% plot resampling scheme
% imshow(resampling);

% define structure for fitting
opt = struct( ...
  'stimulus',    stimulus, ...
  'data',        double(data'), ...
  'model',       {model}, ...
  'metric',      metric, ...
  'dontsave',    {{'testdata', 'modelpred', 'modelfit', 'numiters', 'resnorms'}}, ...
  'resampling',  resampling, ...
  'optimoptions',{{'Display' 'off'}});

if debug
    opt.data = opt.data(:, 1);
end

%
opt

%% Fit the model
mypool = parpool(nworkers);
% fit the model
resultsXVAL = fitnonlinearmodel(opt);
delete(mypool);
%%

% this is the R^2 between the model predictions and the data.
% notice that this cross-validated R^2 is lower than the
% R^2 of the full fit obtained previously.
resultsXVAL.aggregatedtestperformance

%% save

save(fnout, '-struct', 'resultsXVAL');
