% plotDSStreamLines() : Plots stream lines of the learned model.
% plotDSStreamLines(quality) : Plots stream lines of the learned model with
%           the speicifed quality. The possible values for quality are:
%           quality = {'low', 'medium', or 'high'} [default='low']
%
% This function can be only used for 2D models.
%
% Optional input variable -------------------------------------------------
%   o quality: An string defining the quality of the plot. It can have one
%              of the following value: 'low', 'medium', or 'high' [default='low']
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Copyright (c) 2015 Matteo Saveriano, DHRI Lab, TUM,   %%%
%%%          80333 Munich, Germany, http://www.hri.ei.tum.de/en/home/   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The program is free for non-commercial academic use. Please contact the
% author if you are interested in using the software for commercial purposes.
% The software must not be modified or distributed without prior permission
% of the authors. Please acknowledge the authors in any academic publications
% that have made use of this code or part of it. Please use this BibTex
% reference:
% 
%%
function plotDSStreamLines(DSHandle, LRDSHandle, D, quality)

if(nargin<4)
	quality='low';
end

if strcmpi(quality,'high')
    nx=600;
    ny=600;
elseif strcmpi(quality,'medium')
    nx=400;
    ny=400;
else
    nx=25;
    ny=25;
end

ax.XLim = D(1:2);
ax.YLim = D(3:4);
ax_x=linspace(ax.XLim(1),ax.XLim(2),nx); %computing the mesh points along each axis
ax_y=linspace(ax.YLim(1),ax.YLim(2),ny); %computing the mesh points along each axis
[x_tmp, y_tmp]=meshgrid(ax_x,ax_y); %meshing the input domain
x=[x_tmp(:) y_tmp(:)]';

% Compute next modulated position
xd = DSHandle(x); %compute outputs
[D, N] = size(xd);

% Compute reshaping term
if(isempty(LRDSHandle))
    par.xd = zeros(D, N);
else
    par.xd = LRDSHandle(x);
%     truncValues.alphaMin = 0.1;
%     truncValues.rho      = 0.1;
% 
%     alpha       = computeGpWeights(GPR(1), x', truncValues);
%     par.xd(1,:) = alpha' * GPR(1).data.out;
%     par.xd(2,:) = alpha' * GPR(2).data.out;
end

w = ones(2,N);
for i=1:N
    if(norm(x(:,i))<5)
        w(:,i) = [0;0];
    end
    xd(:,i) = xd(:,i) + w(:,i).*par.xd(:,i);
end

streamslice(x_tmp,y_tmp,reshape(xd(1,:),ny,nx),reshape(xd(2,:),ny,nx))
axis([ax.XLim ax.YLim]);box on;