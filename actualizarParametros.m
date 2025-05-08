function parametros_nuevos = actualizarParametros ( parametros, gradientes, alpha)

% ------------------------ ENTRADAS -----------------------------------
% Parametros   : es una estructura con las matrices Wl y los vectores bl de
%                cada una de las capas l=L, ... ,1
% Gradientes   : estructura donde tenemos los gradientes calculados
%                anteriormente: dW1,db1, ....... ,dWL, dbL
% alpha        : es la tasa de aprendizaje

% ------------------------ SALIDAS ------------------------------------
% Parametros_nuevos : estructura que la entrada "parametros" 
%                     pero con los valores actualizados
% ---------------------------------------------------------------------

% Numero total de capas a actualizar
L = length(fieldnames(parametros)) / 2;

% Recorremos las capas para actualizar
for l =1:L
    
    % Extraemos los pesos y sesgos de la capa l
    Wl = parametros.(['W' num2str(l)]);
    bl = parametros.(['b' num2str(l)]);

    % Extraemos los gradientes de la capa l
    dWl = gradientes.(['dW' num2str(l)]);
    dbl = gradientes.(['db' num2str(l)]);


    % Actualizamos los valores según las fórmulas vistas
    Wl = Wl - alpha * dWl;
    bl = bl - alpha * dbl;

    % Guardamos los valores en la nueva estructura
    parametros_nuevos.(['W' num2str(l)]) = Wl;
    parametros_nuevos.(['b' num2str(l)]) = bl;

   

end

end

