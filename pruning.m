function [nuevos_parametros,porcentaje_apagado]=pruning(parametros_entrenados, umbral)

% ----------------------- ENTRADAS ------------------------------
% Parametros_entrenados   : son los parametros entrenados que nos da el
%                           modelo de entrena_DNN
% Umbral                  : el umbral por debajo del cual vamos a podar
% ----------------------- SALIDAS  -----------------------------
% Nuevos_parametros       : son los parametros con el pruning hecho 
% --------------------------------------------------------------

nuevos_parametros = parametros_entrenados;
cantidad_apagadas = 0;
conexiones_totales = 0;

% Determinamos el numero de campos que tiene la estructura. Teniendo en
% cuenta que la estructura tiene por cada capa dos campos.
capas = length(fieldnames(parametros_entrenados))/2;

for i=1:capas

   W = nuevos_parametros.(['W' num2str(i)]);
   [filas, columnas] = size(W);

        for m = 1:filas
            for j = 1:columnas
                if abs(W(m,j)) < umbral
                    W(m,j) = 0;
                    cantidad_apagadas=cantidad_apagadas+1;
                end %end del if
            end %end primer for
        end %end segundo for
   nuevos_parametros.(['W' num2str(i)]) = W;

end %end del for


% Calculamos el porcentaje de conexiones apagadas, para ello primero hay
% que ver el total

for i = 1:capas
    W = parametros_entrenados.(['W' num2str(i)]);
    conexiones_totales = conexiones_totales + numel(W);
end

% Por tanto el porcentaje de conexiones apagadas es
porcentaje_apagado = cantidad_apagadas/conexiones_totales * 100;
end %end function