function gradiente = backPropagation(AL, Y, cache, parametros, derivadaCosto, derivadaActocul, derivadaActsal)

% ------------------------- ENTRADAS -----------------------------------
% AL              : salidas de nuestra red neuronal, en el caso de que tengamos 
%                   m ejemplos de entrada, AL será de dimensiones 1 x m.
% Y               : etiquetas correctas para cada uno de los ejemplos de entrada
% Cache           : estructura que tiene las matrices A y Z de cada una de las 
%                   capas, para acceder a ellas usamos cache.A0, cache.Z0,cache.A1,...
% Parametros      : es otra estructura con los campos W y b de cada capa, para acceder 
%                   a ellos usamos parametros.W1, parametros.b1 ......
% DerivadaCoste   : es la derivada de la función de coste que usemos
% DerivadaActocul : derivada de la función de activación de capas ocultas
% DerivadaActsal  : derivada de la función de activación de capa de salida

% ------------------------- SALIDAS -----------------------------------
% Gradiente       : estructura donde  vamos a guardar los gradientes, y los vamos a notar 
%                   de la siguiente forma: dWL será el gradiente con respecto a la matriz W 
%                   en la capa L, de igual forma tambien tendremos dbL para el gradiente 
%                   respecto del vector b.
% -------------------------------------------------------------------------

% PASO 1 : Dimensiones 
m = size(Y, 2); %numero de columnas de la matriz Y
nCampos = fieldnames(parametros);
L = length(nCampos)/2; %numero de capas contando ocultas y salida, parametros tiene 2 campos por capa

% PASO 2 : Estructura para guardar los gradientes
gradientes = struct();

% PASO 3 : Derivada de la función de coste (Entropia cruzada) respecto a la salida (AL)
dA = derivadaCosto(AL,Y);

ZL = cache.(['Z' num2str(L)]);
A_prev = cache.(['A' num2str(L-1)]);
dZL = dA .* derivadaActsal(ZL);
dWL = (1/m) * dZL * A_prev';
dbL = (1/m) * sum(dZL, 2);
gradientes.(['dW' num2str(L)]) = dWL;
gradientes.(['db' num2str(L)]) = dbL;

% Propagamos hacia atrás: calculamos dA para la capa anterior
WL = parametros.(['W' num2str(L)]);
dA = WL' * dZL;

% PASO 4 : Retropropagación desde la capa L hasta la primera, la variable
% derivada_previa va a tener el gradiente dA^(l) a medida que retrocedamos


for l = L-1:-1:1
        % Recuperar Z de la capa l.
        Zl = cache.(['Z' num2str(l)]);
        % Recuperar A de la capa anterior.
        if l == 1
            A_prev = cache.A0;  % Entrada
        else
            A_prev = cache.(['A' num2str(l-1)]);
        end
        
        % Calcular la derivada de la función de activación:
        % dG = g'(Zl)
        dG = derivadaActocul(Zl);
        
        % Ahora se calcula dZ para la capa l:
        % dZ^{[l]} = dA^{[l]} .* g'(Z^{[l]})
        dZl = dA .* dG;
        
        % Gradientes respecto a los pesos y sesgos:
        % dW^{[l]} = (1/m) * dZ^{[l]} * (A^{[l-1]})'
        % db^{[l]} = (1/m) * sum(dZ^{[l]}, 2)
        dWl = (1/m) * dZl * A_prev';
        dbl = (1/m) * sum(dZl, 2);
        
        % Guardar en la estructura
        gradientes.(['dW' num2str(l)]) = dWl;
        gradientes.(['db' num2str(l)]) = dbl;
        
        % Preparar el gradiente para la siguiente iteración (capa anterior).
        
        Wl = parametros.(['W' num2str(l)]);

        dA = Wl' * dZl;
end


gradiente = gradientes;

end

