function representaPrecision(matriz_de_precisiones,neuronas_capa)

% --------------------  ENTRADAS  --------------------------
% Matriz_de_precisiones  :  es una matriz que tiene como columnas el numero
%                           de capas y como filas el numero de neuronas
%                           por capa
% Neuronas_capa          : es un vector indicando el número de neuronas a
%                          incluir en cada capa
% Esta función representa en un plano 3-D las curvas de nivel de las
% precisiones obtenidas por el modelo.

% Definición de los ejes
  num_capas = 1:size(matriz_de_precisiones,2);
  num_neuronas = neuronas_capa;

% Creacción de la malla original
  [X,Y] = meshgrid(num_capas, num_neuronas);
  Z = matriz_de_precisiones;

% Interpolación sobre la malla
  [Xint, Yint] = meshgrid(linspace(min(num_capas),max(num_capas),100), linspace(min(num_neuronas),max(num_neuronas),100));
  Zint = interp2(X,Y,Z,Xint,Yint,'spline');

% Representación de la superficie 3D
  figure;
  surf(Xint,Yint,Zint);
  shading interp;
  xlabel('Numero de capas ocultas');
  ylabel('Neuronas por capa');
  zlabel('Precisión en %');
  title('Precisión para diferentes dimensiones');
  colorbar;
  view(135,30);

  % Representación del mapa de calor
    figure;
    heatmap(num_capas, num_neuronas, matriz_de_precisiones, ...
    'Colormap', parula, ...
    'XLabel', 'Número de capas ocultas', ...
    'YLabel', 'Neuronas por capa', ...
    'Title', 'Precisión en validación (%)');


end