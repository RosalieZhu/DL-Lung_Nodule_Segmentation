
function [insideC, outsideC] = Chanvese_RegionAverage(phi, inputImg)
% inside C와 outside C를 계산해준다.
% Compute c1 and c2 as used in the Chan-Vese segmentation algorithm.
% c1 and c2 are given by
% c1 = integral(u0*H(phi))dxdy/integral(H(phi))dxdy
% c2 = integral(u0*(1-H(phi))dxdy/integral(1-H(phi))dxdy


L = inputImg;

inidx = phi >= 0;
outidx = phi < 0;

indata = L(inidx);
outdata = L(outidx);
sum1 = sum(indata);
sum2 = sum(outdata);

insideC = double(sum1)./length(indata);
outsideC = double(sum2)./length(outdata);

if isnan(insideC)
    insideC = 0;
end
if isnan(outsideC)
    outsideC = 0;
end
