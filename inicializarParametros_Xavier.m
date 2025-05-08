function parametros = inicializarParametros_Xavier(dimensiones)

% ---------------------- ENTRADAS -----------------------------
% Dimensiones  : vector que contiene el numero de neuronas por cada capa, y 
%                su longitud es el número de capas que tenemos.
% ---------------------- SALIDAS ------------------------------
% Parámetros   : estructura que contiene: W1,b1,W2,b2, ....... que son los 
%                correspondientes W y b de la matriz de pesos y vector de sesgos 
%                de cada capa
% --------------------------------------------------------------------------

% Numero de capas, incluidas la de entrada y salida
L= length(dimensiones)-1;

%Creamos una estructura vacía para ir rellenándola
parametros=struct();

% -----------------------------------------------
% INICIALIZACIÓN CON NÚMEROS ALEATORIOS
% -----------------------------------------------

for l = 1:L
        % Inicialización Xavier (normal):
        % Se extraen valores de N(0, 1/n(l)), donde n(l) es el número de
        % neuronas en la capa anterior.
        parametros.(['W' num2str(l)]) = randn(dimensiones(l+1), dimensiones(l)) * sqrt(2 /(dimensiones(l)+dimensiones(l+1)) );
        
        % Inicializamos los sesgos a 0
        parametros.(['b' num2str(l)]) = zeros(dimensiones(l+1), 1);
end %end del for

end