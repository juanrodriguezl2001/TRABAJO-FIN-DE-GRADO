% PROGRAMA PRINCIPAL

% ---------------------------------
% FUNCIÓN DE ACTIVACIÓN A USAR
% ----------------------------------
function A = relu(Z)
    A = max(0, Z);
end

function A = sigmoid(Z)
    A = 1 ./ (1 + exp(-Z));
end
% ---------------------------------
% PREPARACIÓN DE LOS DATOS
% ----------------------------------

% Numero de muestras por señal
fs=200;
% Numero de ejemplos
N=16000;
% Potencia de ruido
Pot_ruido = 1.3;
% Impulsividad de ruido
Imp_ruido = 3;




% -------------------- DATOS DE ENTRENAMIENTO --------------------
[Datos_entrenamiento, Etiquetas_entrenamiento] = genera_datos_ECG(N,fs,Pot_ruido,Imp_ruido);

Datos_entrenamiento = Datos_entrenamiento';
Etiquetas_entrenamiento = Etiquetas_entrenamiento';

% -------------------- DATOS DE VALIDACIÓN --------------------------------
[Datos_val, Etiquetas_val ] = genera_datos_ECG(N*0.2,fs,Pot_ruido+0.3,Imp_ruido-0.6);

Datos_val = Datos_val';
Etiquetas_val = Etiquetas_val';

% -------------------- DATOS DE TEST --------------------------------
[Datos_test, Etiquetas_test ] = genera_datos_ECG(N*0.2,fs,Pot_ruido+0.46,Imp_ruido-0.9);

Datos_test = Datos_test';
Etiquetas_test = Etiquetas_test';



% --------------------------------
% DEFINICIÓN DE LA ARQUITECTURA 
% --------------------------------
% Definimos aqui las capas que va a tener nuestra DNN
% esto significa que tenemos 4 capas, la primera tiene 10 neuronas, la
% segunda 5, la tercera 3 y la capa final una sola neurona (clasificación
% binaria).

% Numero de columnas de nuestros datos de entrenamiento
neuronas_entrada = size(Datos_entrenamiento,1);
    % % -----------------  GRID SEARCH  -----------------------------
    % % (descomentar si se está buscando las dimensiones óptimas)
    % num_capas_ocultas = 1:4;
    % neuronas_por_capa = [25,50,100,150];
    % % Matriz para almacenar la precisión de cada dimensión
    % resultados_precision = zeros(length(num_capas_ocultas),length(neuronas_por_capa));

    % ----------------  DIMENSIONES FIJAS -------------------------
    num_capas_ocultas = 2;
    neuronas_por_capa = 25;

% Parámetros para la simulación
  alpha = 0.1;
  nIteraciones = 1000;

% Comenzamos el bucle de entrenamiento, en el caso de que solo haya una
% dimensión fijada no hay problema
for i= 1:length(num_capas_ocultas)
    for j = 1:length(neuronas_por_capa)

  num_capas = num_capas_ocultas(i);
  num_neuronas= neuronas_por_capa(j);
  dimensiones=[neuronas_entrada, repmat(num_neuronas,1,num_capas),1];


% PASO 1: Entrenamos nuestro modelo con los datos de entrenamiento y validación  
[parametros_entrenados, vectorCoste_training] =entrena_DNN(Datos_entrenamiento,Etiquetas_entrenamiento,Datos_val,Etiquetas_val,dimensiones,@relu, @sigmoid, alpha, nIteraciones);

% % PASO 2 : Hacemos el pruning a los parámetros
% % (descomentar en el caso de que se quiera usar la técnica de pruning)
% umbral = 0.05;
% [nuevos_parametros,porcentaje_apagado]=pruning(parametros_entrenados,umbral);
% fprintf('El porcentaje de conexiones apagadas es el %.2f%% \n',porcentaje_apagado);

% PASO 3 : Le aplicamos ahora el forwardPropagation para ver las salidas
% (en el caso de usar pruning, sustituir parametros_entreador por
%  nuevos_parametros)
[AL_test, ~] = forwardPropagation(Datos_test, parametros_entrenados, @relu, @sigmoid);

% PASO 4 : Como estamos haciendo clasificación binaria, si el resultado es
%          mayor que 0.5 deberíamos interpretarlo como un 1, y si no como un 0
predicciones = AL_test > 0.5;

% PASO 5 : Cálculo de la precisión
numero_pruebas = size(Etiquetas_val,2);
aciertos = sum(predicciones == Etiquetas_test);
precision = aciertos / numero_pruebas;

% -------------------------------------------------
%           ANALISIS DE ERRORES
% -------------------------------------------------
cadDims = sprintf('%d ', dimensiones);  

fprintf('Dimensiones [%s], precisión: %.2f%%\n',cadDims ,precision*100);

% % Guardamos la precision de estas dimensiones 
% % (descomentar en el caso de que estemos buscando las dimensiones óptimas)
% resultados_precision(i,j)=precision*100;
    
    end 
end

% %Representación de la superficie de precisión/dimensiones
% %(descomentar en el caso de buscar las dimensiones óptimas)
%representaPrecision(resultados_precision, neuronas_por_capa);
