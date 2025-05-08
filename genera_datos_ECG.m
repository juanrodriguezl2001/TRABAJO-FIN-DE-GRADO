function [X, y] = genera_datos_ECG(N, fs, Pot_ruido, Imp_ruido)

% ---------------------- ENTRADAS --------------------------------
% N         : Número de señales que deseamos generar en total
% fs        : Frecuencia de muestreo, indica el número de muestras por segundo
% Tmax      : Tiempo de cada señal en segundos 
% Pot_ruido : potencia del ruido generado
% Imp_ruido : impulsividad del ruido, es un valor que controla como de
%             impulsivo es el ruido generado. Un valor alto implica ruido
%             mas impulsivo.
% --------------------- SALIDAS -------------------------------------
% X         : Matriz con las señales generadas, dimensiones N x fs
% Y         : Etiquetas de cada una de las señales
% ------------------------------------------------------------------

Tmax = 1;

% SEÑAL ECG + RUIDO --> ETIQUETA '0'
% SEÑAL PURAMENTE RUIDOSA --> ETIQUETA '1'

% --------------------  Inicialización de matrices. ----------------------
% La matriz X tiene por filas los ejemplos que tenemos y por columnas cada
% una de las "características". 
% La matriz Y tiene una sola columna, que son las etiquetas, y tantas filas
% como ejemplos tengamos.

X = zeros(N, fs*Tmax);
y = zeros(N, 1);

t = (0:(fs*Tmax-1))/fs;

% Generar datos con distribución Bernoulli (p=0.5)
for i = 1:N
    etiqueta = rand() < 0.5; % Bernoulli con p=0.5
    if etiqueta == 0
        ecg_signal = FUN_ECG(fs, Tmax);
        noise_signal = FUN_NOISE(t, Pot_ruido, Imp_ruido);
        signal = ecg_signal + noise_signal;
        
    else
        signal = FUN_NOISE(t, Pot_ruido, Imp_ruido);
        
    end %end del if
 
    X(i, :) = signal;
    y(i) = etiqueta;

end %end del for

end