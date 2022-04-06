function L = visualJacobianMatrix(s, depths, KP)
    px = KP(1,1);
    py = KP(2,2);
    c0 = [KP(1,3), KP(2,3)];
    
    L = [];

    for i=1:4
        x = (s(1,i) - c0(1)) / px; 
        y = (s(2,i) - c0(2)) / py; 
        
        L_i = L_matrix(x, y, depths(i));
        L_i = - diag([px, py]) * L_i;
        
        L = [L; L_i];
    end
end

function L = L_matrix(x, y, z)
    L =  [ 1/z, 0,   -x/z, -x*y,     (1+x*x), -y; ...
             0,   1/z, -y/z, -(1+y*y), x*y,    x ];
end