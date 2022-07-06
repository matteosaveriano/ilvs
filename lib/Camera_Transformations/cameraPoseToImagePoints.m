% Transform a given camera pose into 4 points on the image plane
%
% NOTE: orientation is expressed in Euler angles or rotation matrix
% (X-Y-Z around the inertial frame of reference)

function [points, depth] = cameraPoseToImagePoints(pose, initial3DPts, KP)
    if nargin < 3
        KP = zeros(3,4);
        KP(1:3,1:3) = eye(3,3);
    end
    % rotation matrix R_WCS_CCS
    if all(size(pose)==[4, 4]) % 'pose' is an homogeneous transformation
        R = pose(1:3, 1:3);
        r = pose(1:3, 4);
    else
        R = cameraRotationMatrix(pose(4), pose(5), pose(6));
        r = pose(1:3);
    end
    
    % 3D coordinates of the 4 points wrt the NEW camera frame
    A3D_cam = R'*(initial3DPts(:,1)-r);
    B3D_cam = R'*(initial3DPts(:,2)-r);
    C3D_cam = R'*(initial3DPts(:,3)-r);
    D3D_cam = R'*(initial3DPts(:,4)-r);
    
    depth = [A3D_cam(3), B3D_cam(3), C3D_cam(3), D3D_cam(3)];
    
    % Homogeneous coordinates of the points on the image
    A3D_hom = KP * [A3D_cam; 1];
    B3D_hom = KP * [B3D_cam; 1];
    C3D_hom = KP * [C3D_cam; 1];
    D3D_hom = KP * [D3D_cam; 1];

    % NEW projections
    points = zeros(2,4);
    points(:,1) = A3D_hom(1:2)/A3D_hom(3);
    points(:,2) = B3D_hom(1:2)/B3D_hom(3);
    points(:,3) = C3D_hom(1:2)/C3D_hom(3);
    points(:,4) = D3D_hom(1:2)/D3D_hom(3);
end