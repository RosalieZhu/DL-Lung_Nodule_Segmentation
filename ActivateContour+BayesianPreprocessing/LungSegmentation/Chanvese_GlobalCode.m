
function seg = Chanvese_GlobalCode(inputImg)

%%% This code is made by BAMI LAB %%%
%%% This link is a site, we refer to make Chanvese. %%%
%%% https://kr.mathworks.com/matlabcentral/fileexchange/23445-chan-vese-active-contours-without-edges %%%
%------------------------------- Chanvese Setting ------------------------
num_iter = 100;
mu = 0.001;

%------------------------------- Chanvese -------------------------------
L = im2double(inputImg)*255;
phi0 = Chanvese_InitPhi(inputImg);

%------------------------------- Main Loop -------------------------------
for n=1:num_iter
    [c1, c2] = Chanvese_RegionAverage(phi0, L);
    
    force_image = -(L-c1).^2 + (L-c2).^2;
    force = mu*Chanvese_Kappa(phi0)./max(max(abs(Chanvese_Kappa(phi0)))) + force_image;
    force = force./(max(max(abs(force))));
    
    dt = 0.5;
    phi0 = phi0 + dt.*force;
end

seg = phi0; %-- Get mask from levelset