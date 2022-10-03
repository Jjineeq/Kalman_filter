clear all

Nsamples = 500;
Xsaved   = zeros(Nsamples, 1);
Xsaved1  = zeros(Nsamples, 1);
Xmsaved  = zeros(Nsamples, 1);
Xmsaved1  = zeros(Nsamples, 1);


for k=1:Nsamples
  xm = GetSonar();
  xm1 = GetSonar();
  x  = LPF(xm);
  x1  = LPF1(xm);

  Xsaved(k)  = x;
  Xsaved1(k)  = x1;
  Xmsaved(k) = xm;
  Xmsaved1(k) = xm1;
end


dt = 0.02;
t  = 0:dt:Nsamples*dt-dt;

figure
hold on
plot(t, Xmsaved, 'r.');
plot(t, Xmsaved1, 'k');
plot(t, Xsaved, 'b');
%legend('Measured', 'LPF')
