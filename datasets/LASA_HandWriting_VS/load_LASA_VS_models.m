% A matlab functions to load samples from the LASA HandWritten dataset
%
% Author: Matteo Saveriano
% Date:   04.11.2015
%
% Please acknowledge the authors in any academic publications
% that have made use of this library by citing the following paper:
%
%  S. M. Khansari-Zadeh and A. Billard, "Learning Stable Non-Linear Dynamical 
%  Systems with Gaussian Mixture Models", IEEE Transaction on Robotics, 2011.
%
% To get latest upadate of the software please visit
%                          http://lasa.epfl.ch/khansari
%

function [demoVS, name] = load_LASA_VS_models(datasetPath, modelNumber)
    names = {'Angle','BendedLine','CShape','GShape', 'heee',...
             'JShape','Khamesh','LShape','NShape','PShape',...
             'RShape','Saeghe','Sharpc','Sine','Snake',...
             'Spoon','Sshape','Trapezoid','Worm','WShape','Zshape',...
             'JShape_2', 'Leaf_1', 'Leaf_2', 'DoubleBendedLine','Line',...
             'Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4'};
         
    if(modelNumber<1 || modelNumber>30)
        disp('Wrong model number!')
        error('Please try again and type a number between 1-30.')
    end
    
    tmp = load([datasetPath 'DataSet/' names{modelNumber} '_VS']); %loading the model
    
    demoVS = tmp.demoVS;
    
    name = names{modelNumber};
end
