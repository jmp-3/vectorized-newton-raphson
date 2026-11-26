clear; close all; clc;
fprintf("\n====== comeco ======\n")


G = zeros(3,3);


B = [ -12.5,  4.16,  8.33;
     4.16, -24.16, 20;
     8.33 ,  20, -28.33]


Y = G + 1i.*B


Sb = 50; %MVA
Vb = 120; %kV
           %  1        2       3
V       =   [1,       nan,      nan]
theta   =   [0,       nan,      nan]
P       =   [nan,     1.5,    -1]
Q      =    [nan,      0.057,    -1]


v_2 = 1
v_3 = 1
theta_2 = 0
theta_3 = 0


Pd = [1,     0,      1]       % carga ativa por barra (pu)
Qd = [0.5,   0,      1]       % carga reativa por barra (pu)
plot_j = []
plot_p = []
plot_q = []


for i=1:30


   fprintf("\n====== iteracao numero: %d======\n", i)
   V = [1, v_2, v_3];
   theta = [0, theta_2, theta_3];
 
  % ======= A partir daqui eh generico ==========
   Vkl = V.*V';                            % gerar matrizes que combinam as tensoes
                                           % multiplicando as tensoes corresponden-
                                           % tes ao subindice da matriz


   Vkk = Vkl.*eye(size(Vkl));              % para finalidade de calculos eh interes-
                                           % sante ter essa matriz com apenas os qua-
                                           % drados na diagonal


   Vkl = Vkl.*~eye(size(Vkl));             % para o calculo das potencias nao se con-
                                           % sidera os elementos da diagonal


   theta_kl = theta - theta';              % geracao de uma matriz que subtrai os angu-
                                           % los correspondente aos subindices


   sin_theta_kl = sin(theta_kl);           % gerar matriz dos senos
   cos_theta_kl = cos(theta_kl);           % gerar matriz dos cosenos


   % Calculo dos delta p
   delta_p = P - sum(Vkl.*(G.*cos_theta_kl + B.*sin_theta_kl) + Vkk.*G)
   delta_q = Q - sum(Vkl.*(G.*sin_theta_kl - B.*cos_theta_kl) - Vkk.*B)
  
   % Criar matrizes e variaveis uteis
   n = size(Vkl);
   Xkk = eye(n);
   Xkm = ~eye(n);          % fora da diagonal (k ~= m)


   Vkl = V.*V';            % Para o calculo dos valores na jacobiana
                           % se consideram os da diagonal

   sin_km = sin_theta_kl'; % elemento (k,m) = sin(theta_k - theta_m)
   cos_km = cos_theta_kl'; % elemento (k,m) = cos(theta_k - theta_m)


   % Calculo da jacobiana
   % |-------------------------- FORMULA --------------------------| ---- ESTRUTURA ----|
   % dp/dTheta
   Hkk = (-B.*Vkk - sum(Vkl.*(G.*sin_theta_kl - B.*cos_theta_kl)))          .*Xkk;
   Hkm = Vkl.*(G.*sin_km - B.*cos_km)                                       .*Xkm;
   H = Hkm + Hkk;


   % dp/dV
   Nkk = ((G.*V) + sum(V'.*(G.*cos_theta_kl + B.*sin_theta_kl)))            .*Xkk;
   Nkm = V'.*(G.*cos_km + B.*sin_km)                                        .*Xkm;
   N = Nkm + Nkk;


   % dQ/dTheta
   Mkk = (-G.*Vkk  + sum(Vkl.*(G.*cos_theta_kl + B.*sin_theta_kl)))         .*Xkk;
   Mkm = -Vkl.*(G.*cos_km + B.*sin_km)                                      .*Xkm;
   M = Mkm + Mkk;


   % dQ/dV
   Lkk = (-B.*V + sum(V'.*(G.*sin_theta_kl - B.*cos_theta_kl)))             .*Xkk;
   Lkm = V'.*(G.*sin_km - B.*cos_km)                                        .*Xkm;
   L = Lkm + Lkk;


   % ======= A partir daqui nao eh mais generico ==========




   jac =  [    H(2,2), H(2,3), N(2,2), N(2,3);
               H(3,2), H(3,3), N(3,2), N(3,3);
               M(2,2), M(2,3), L(2,2), L(2,3);
               M(3,2), M(3,3), L(3,2), L(3,3)]


   plot_j = [plot_j; reshape(jac.', 1, [])];         % vetores apenas para plotagem
   plot_p = [plot_p; delta_p];
   plot_q = [plot_q; delta_q];

   % parar quando convergir
   if max(abs([delta_p(2:3), delta_q(2:3)])) < 1e-8, break; end


   inv_jac = -pinv(jac);               % apenas para visualizacao da
                                       % inversa, se desejado


   F = -[delta_p(2), delta_p(3), delta_q(2), delta_q(3)]; 
   X = [theta_2, theta_3,  v_2, v_3];


   X = X' - pinv(jac)*F';  % como os vetores estao como vetor-linha prec-
                           % isamos transpor se formos usar a formula padrao


   X = X';                 % e depois transpor de volta para ficar
                           % como o resto do programa
                          


   theta_2 = X(1);         % atualizar
   theta_3 = X(2);
   theta_2_deg = X(1) * 180/pi % visualizacao apenas
   theta_3_deg = X(2) * 180/pi % visualizacao apenas
   v_2 = X(3)
   v_3 = X(4)


end


   % plotar figuras de convergencias


   figure(1)
   keys = {"delta P1", "delta P2", "delta P3"};
   plot (plot_p, 'linewidth', 2);
   h = legend(keys,"location", "east");
   h = title('Convergência Delta P');
   xlabel ("n iterações");


   figure(3)
   keys = {"delta Q1", "delta Q2", "delta Q3"};
   plot (plot_q, 'linewidth', 2);
   h = legend(keys,"location", "east");
   h = title('Convergência Delta Q');
   xlabel ("n iterações");


   figure(2)
   keys = {"H22","H23","N22","N23","H32","H33","N32","N33","M22","M23","L22","L23","M32","M33","L32","L33"};
   plot (plot_j, 'linewidth', 2);
   h = legend(keys,"location", "east");
   h = title('Parametros da matriz jacobiana');
   xlabel ("n iterações");


   % calculo das potencias

   V = [1, v_2, v_3];          % usar os valores da ultima atualizacao
   theta = [0, theta_2, theta_3];

   Vkl = V.*V';
   Vkk = Vkl.*eye(size(Vkl));
   Vkl = Vkl.*~eye(size(Vkl));
   theta_kl = theta - theta';
   sin_theta_kl = sin(theta_kl);
   cos_theta_kl = cos(theta_kl);
  
   disp('pu')
   Pcalc =  sum(Vkl.*(G.*cos_theta_kl + B.*sin_theta_kl) + Vkk.*G)
   Qcalc =  sum(Vkl.*(G.*sin_theta_kl - B.*cos_theta_kl) - Vkk.*B)
   V = V


   Pg = Pcalc + Pd             % potencia gerada = injecao liquida + carga
   Qg = Qcalc + Qd

   disp('MVA e kV')
   Pcalc = Pcalc * Sb
   Qcalc = Qcalc * Sb
   V = V * Vb
   Pg = Pg * Sb
   Qg = Qg * Sb




fprintf("\n====== fim ======\n")
