function xlpf1 = LPF1(x)
%
%
persistent prevX
persistent firstRun


if isempty(firstRun)
  prevX = x;
  
  firstRun = 1;  
end


alpha = 0.9;
xlpf1 = alpha*prevX + (1 - alpha)*x;

prevX = xlpf1;