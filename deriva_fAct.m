function [der_fAct]=deriva_fAct(fAct)

% ------------------- ENTRADAS -------------------------------
% fAct     : función de activación de la cual se quiere la derivada
% ------------------- SALIDAS --------------------------------
% der_fAct : derivada de la función de activación de entrada
% ------------------------------------------------------------

nombre = func2str(fAct);  % Convierte la función en cadena de texto

    if strcmp(nombre, 'relu')
        der_fAct = @derivadaReLU;
    elseif strcmp(nombre, 'sigmoid')
        der_fAct = @derivadaSigmoid;
    else
        error(['Función de activación no soportada: ' nombre]);
    end

end


% --------------  FUNCIONES AUXILIARES -------------------------------
function A = relu(Z)
    A = max(0, Z);
end

function dA = derivadaReLU(Z)
    dA = Z > 0;
end

function A = sigmoid(Z)
    A = 1 ./ (1 + exp(-Z));
end

function dA = derivadaSigmoid(Z)
    A = sigmoid(Z);
    dA = A .* (1 - A);
end
