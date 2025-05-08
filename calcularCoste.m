function coste = calcularCoste(AL, Y)

% ----------------------- ENTRADAS -----------------------------------
% AL : variable que tiene la salida de la red neuronal, en el caso de
%      meterle m ejemplos, es un vector 1 x m.
% Y  : vector con las etiquetas (valores reales) de los datos de
%      entrada, Y(1) es la salida correcta y debe compararse con AL(1)
% ----------------------- SALIDAS -----------------------------------
% Coste : es el coste MSE calculado según las entradas
% -------------------------------------------------------------------


% Número de ejemplos
m= size(Y,2);  % devuelve el número de columnas de la matriz Y, si fuera '1' nos daria el numero de filas 

% Calculamos la entropia cruzada
epsilon=1e-8;
coste = -(1 / m) * sum( Y .* log(AL + epsilon) + (1 - Y) .* log(1 - AL + epsilon) , 'all');


end 