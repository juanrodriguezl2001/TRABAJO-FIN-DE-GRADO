% ALGORITMO DE FORWARD PROPAGATION
function [AL, cache]= forwardPropagation(X, parametros, funcionActivacion_ocultas, funcionActivacion_salida)

% -------------------  ENTRADAS  ------------------------------------------
% X                 : Datos de entrada (caract de entrada a la primera capa)
% parametros        : Estructura con los pesos (W) y sesgos (b)
% funcionActivacion : Función de activación para la DNN
%  ------------------  SALIDAS  ---------------------------------------------
% AL                : Predicción de nuestra DNN
% cache             : Estructura con valores intermedios (Z,A)

% ----------------------------------------------------
%  PREPARACIÓN DE LA ESTRUCTURA CACHE
% ----------------------------------------------------
cache= struct();
cache.A0 = X; 
% La activación de la capa 0 son los datos de entrada

% ---------------------------------------------------
%  CÁLCULO DE NUMERO TOTAL DE CAPAS
% ---------------------------------------------------
nCampos=fieldnames(parametros);
L = length(nCampos) / 2;
% Esto se debe a que cada capa tiene 2 campos (W y b) por eso /2

% --------------------------------------------------
% FORWARD PROPAGATION
% --------------------------------------------------
Act_prev = X; %La activación previa es la entrada
for l = 1:L-1
    
    % Paso 1 : Extraer W y b de la capa 'l'
    W_l = parametros.(['W' num2str(l)]);
    b_l = parametros.(['b' num2str(l)]);

    % Paso 2 : Calcular la salida Z^(l)
    Z_l = W_l * Act_prev + b_l;

    % Paso 3 : Guardamos la matriz Z_l en cache
    cache.(['Z' num2str(l)]) = Z_l;

    % Paso 4 : Aplicamos la función de activación para calcular A=g(Z)
    A_l=funcionActivacion_ocultas(Z_l);

    % Paso 5 : Guardamos la variable A_l en cache
    cache.(['A' num2str(l)]) = A_l;

    % Paso 6 : Actualizamos la activación para el siguiente paso
    Act_prev = A_l;

end

% Capa de salida

W_L = parametros.(['W' num2str(L)]);
b_L = parametros.(['b' num2str(L)]);

% Paso 2: Calcular Z^L
Z_L = W_L * Act_prev + b_L;

% Paso 3: Guardar Z^L en cache
cache.(['Z' num2str(L)]) = Z_L;

% Paso 4: Aplicar sigmoide como función de activación para la capa de salida
AL = funcionActivacion_salida(Z_L);

% Paso 5: Guardar AL en cache
cache.(['A' num2str(L)]) = AL;
end