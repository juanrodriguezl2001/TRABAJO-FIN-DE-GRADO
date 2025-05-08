function [ parametros, Vector_coste]=entrena_DNN(X_trai,Y_trai,X_vali,Y_vali,dimensiones,fActocul,fActsal,alpha0,numIteraciones)

% ------------------------------------
% INICIALIZACIÓN DE PARÁMETROS
% ------------------------------------
parametros = inicializarParametros_Xavier(dimensiones);

% ------------------------------------
% INICIALIZACIÓN DE VECTORES DE COSTE
% ------------------------------------
Vector_coste = zeros(1, numIteraciones);
Vector_coste_vali = zeros(1, numIteraciones);

% ------------------------------------
% INICIALIZACIÓN EARLY STOPPING
% ------------------------------------
% (comentar en el caso de que no se quiera usar early stopping)
Iteraciones_maximas = 10;           % numero de iteraciones sin mejorar 
coste_anterior = inf;               % guardamos el coste anterior para comparar
contador = 0;                       % contador de iteraciones
parametros_mejores = parametros;    % los mejores parámetros, por si empeoramos

% ------------------------------------
%  EXPONENTIAL DECAY
% ------------------------------------
 % (en el caso de que no se quiera usar Learning Rate Scheduler, poner k=0)
k=0.001;

% -----------------------------------
% BUCLE DE ENTRENAMIENTO
% -----------------------------------

% Declaramos el numero de iteraciones y la tasa de aprendizaje de nuestro
% modelo

for i = 1:numIteraciones

 % PASO 1 : Hacemos el ForwardPropagation
 [AL_trai, cache] = forwardPropagation(X_trai, parametros,fActocul,fActsal);
 [AL_vali, ~] = forwardPropagation(X_vali, parametros,fActocul,fActsal);

 % PASO 2 : Calculamos el coste 
 coste_trai = calcularCoste(AL_trai, Y_trai);
 coste_vali = calcularCoste(AL_vali, Y_vali);
 
    % Guardamos el coste en un vector para luego hacer una gráfica
      Vector_coste(i)= coste_trai;
      Vector_coste_vali(i)=coste_vali;
 
 % PASO 3 : Algoritmo BackPropagation
    % Llamamos a la funcion derivada_fAct para calcular las derivadas
    derivada_fActOcul = deriva_fAct(fActocul);
    derivada_fActSal = deriva_fAct(fActsal);
    gradientes = backPropagation (AL_trai, Y_trai, cache, parametros, @derivadaCostoCE, derivada_fActOcul,derivada_fActSal);
 
 % PASO 4 : Actualizamos los parámetros de nuestra red
 % ---------------------------- EXPONENTIAL DECAY ------------------------
 %    % Actualizamos el valor de alpha antes de actualizar parámetros
      alpha = alpha0 * exp(-k * i);
  
  parametros_nuevos = actualizarParametros ( parametros, gradientes, alpha);
  parametros = parametros_nuevos;

 % PASO 5: EARLY STOPPING
 % (comentar en el caso de no querer usar Early Stopping)
 if coste_vali < coste_anterior - 1e-6
     coste_anterior = coste_vali;
     parametros_mejores = parametros;
     contador = 0;
 else 
     contador = contador +1;
 end   % end del if

if contador >= Iteraciones_maximas
    fprintf(' Denteniendo el entrenamiento en la iteración %d por early stopping\n',i);
    break;
end
% -------------------------------------------------------------------------------

end    % end del for 

% Una vez hemos salido del bucle de entrenamiento, los parametros serán los
% mejores que hayamos guardado
% (comentar esta línea en el caso de no querer usar Early Stopping)
parametros = parametros_mejores;

% ----------------------------------
% VISUALIZACIÓN DE LOS RESULTADOS
% ----------------------------------

% Mostramos la gráfica con la evolución de los costes
figure;
plot(Vector_coste(1:i), 'b'); hold on;
plot(Vector_coste_vali(1:i), 'r');
xlabel('Iteración');
ylabel('Coste');
title('Evolución del coste');
legend('Entrenamiento', 'Validación');
grid on;             

end

% --------------- FUNCIONES AUXILIARES ----------------------------------

function dAL = derivadaCostoCE(AL, Y)
    % Agregamos un pequeño epsilon para evitar divisiones por cero
    epsilon = 1e-8;
    dAL = - (Y ./ (AL + epsilon) - (1 - Y) ./ (1 - AL + epsilon));
end


