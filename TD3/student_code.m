clear all
close all

%%%%%%%%% Exercise 1: compute entropy of 2 neurons, with varying covariance matrices  %%%%%%%

% Calculo da entropia de duas variaveis aleatorias para diferentes matrizes
% de covariancia.
% Sigma = matriz de covariancia.
for i = 1:3
    
if i==1
     Sigma = [1 0; 0 1];
elseif i==2
    Sigma = [1 0.5; 0.5 1];
elseif i==3
    Sigma = [1 0.99; 0.99 1];
end
    % Entropia da distribuicao normal multivariada
    H = 0.5*(log(2*pi*exp(1)*det(Sigma )))
end

% Quando as variaveis estao muito correlacionadas, conhecendo a variavel 1,
% voce conhece a variavel 2. Dessa forma, a entropia (incerteza) dele eh
% muito baixa. É possivel ver isso na figura da direita do slide 7, em que
% a correlacao eh de 0.99. Quando as variaveis nao estao nada
% correlacionadas, nao da pra saber nada da variavel 2 a partir da variavel
% 1. Assim, a entropia (incerteza) eh mt alta.

%%  Load data, x: 16000 image patches of size 12 x 12 = 144

load('data.mat')
% x: 16000 image patches of size 12 x 12 = 144
% X: 1 full image 
% 
% WZ: decorrelating filters (filtros decorrelacionados)
% Matriz de pesos que eh usada para decorrelacionar os dados de entrada.
% Essa matriz eh calculada a partir da matriz de covariancia dos dados e
% serve para transformar o conjunto de dados original em um novo conjunto
% de dados no qual as variaveis sao decorrelacionadas.
% Correlacao: medida que indica a relacao entre duas variaveis.Quando duas
% variaveis estao positivamente correlacionadas, isso significa que elas
% tendem a aumentar ou diminuir juntas. Quando a correlacao eh zero, nao ha
% relacao linear entre as duas variaveis.

% WI: ICA filters (filtros de Analise de Componentes Independentes)
% Matriz de pesos para realizar a analise de componentes independentes (ICA)
% nos dados de entrada decorrelacionados. Essa analise (ICA) busca
% encontrar componentes independentes em um conjunto de sinais misturados.
% Essa tecnica eh usada para separar as diferentes fontes de sinais
% misturados. Ele eh usado para extrair informacoes uteis de um sinal
% composto por varias fontes independentes. Assim, esse filtro busca
% separar os sinais independentes em componentes independentes e
% identificar as informacoes uteis que cada componente representa.

%% %%%%%%%%%%% Exercise 2 Compute the covariance of x 

% Matriz de covariancia de x (patches de imagem)
C  = x*x'/size(x, 2); % covariance of x

% Convolucao do patches de imagem com o filtro decorrelacionado para gerar
% a resposta z. O filtro WZ vai representar a nossa retina. x sao os fotons
% que chegam na retina e z sao as respostas da retina.
z = WZ*x;

% Matriz de covariancia das respostas do filtro de decorrelacao
Czz = z*z'/size(z, 2); % covariance of z

% Covariancia do estimulo (x) e a covariancia da resposta, ou seja, a
% covariancia do sinal filtrado (z).
figure('Name', 'covariance')
subplot(1, 2, 1)
imagesc(C(1:12, 1:12)); 
colormap('gray')
title('stimulus covariance')
subplot(1, 2, 2)
imagesc(Czz(1:12, 1:12)); 
colormap('gray')
title('response covariance')

% A primeira imagem representa a matriz de covariancia dos estimulos, em
% que o preto representa 0, o branco valores representam 1 e o
% cinza valores entre 0 e 1. A segunda imagem representa a matriz de
% covariancia das resposta de um conjunto de filtros decorrelacionados
% aplicados aos estimulos.
% Os filtros decorrelacionam os estimulos, tornando a covariancia das
% respostas mais dispersas. Como a covariancia indica o grau de
% interdependencia linear entre duas variaveis, essa covariancia fica mais
% dispera após a filtragem, já que as respostas ficam com maior
% interdependencia. Assim, as respostas filtradas possuem apenas
% valores 1 ou 0.

%% Exercise 3 - convolve images with filters

% O filtro WZ eh redimensionado para uma matriz 12x12
% reshape 1 filter
w = reshape(WZ(78, :), 12, 12);

% Convolucao dos estimulos com o filtro WZ
Z = conv2(X, w, 'same');

figure('Name', 'images')
subplot(2, 2, 1)
imagesc(X);
subplot(2, 2, 2)
imagesc(Z, [-1 1]);

% Adiciona um ruido gaussiano nos estimulos.
Xn = X + 0.7*std(X(:))*randn(size(X));

% Convolucao dos estimulos com ruido e o filtro WZ.
Zn = conv2(Xn, w, 'same');

% Imagem original e a imagem após a convolucao.
% Em baixo: imagem com ruido e imagem ruidosa após convolucao.
subplot(2, 2, 3)
imagesc(Xn);
subplot(2, 2, 4)
imagesc(Zn, [-1 1]);
colormap('gray')

% Após filtrar a imagem original (X), percebe-se que o filtro realçou
% as caracteristicas importantes de uma imagem. A adição de ruido na
% imagem original afeta negativamente a qualidade da imagem resultante.
% Portanto o filtro eh incapaz de realçar as caracteristicas importantes da
% imagem, além do filtro aumentar o ruido.

%% Exercise 4: denoising filter

% Ruido gaussiano em diferentes niveis
eta = [0 0.3 1.4]*std(x(:));
figure('Name', 'filters at different noise level')

% loop over different noise levels
for i = 1:3
    % Filtro para o ruido gaussiano
    % Wn  = inv( <xn*xn> )*<xxn'>
    Wn = (C + eta(i)^2*eye(144))\C;

    % Filtro que eh um combinacao do filtro WZ para extrair os principais
    % componentes da resposta dos neuronios ao estimulo visual e o filtro
    % do ruido gaussiano Wn que eh construido com base na matriz de
    % covariancia do estimulo visual e no nivel de ruido adicionado.
    Wcombined = WZ*Wn;

    wcombined = reshape(Wcombined(:, 78), 12, 12);
    
    % Imagem plota o filtro combinado para diferentes niveis de ruido.
    subplot(1, 3, i)
    imagesc(wcombined); axis square; axis off;
end

% A imagem da esquerda eh o filtro WZ original, ou seja, sem o filtro para
% o ruido gaussiano. Logo, há um grande contraste e, assim, há um aumento
% do ruido. À medida que aumentamos "eta", estamos aumentando a quantidade
% de pixels que vao ser usados pelo filtro do ruido. Como pode ser visto
% na figura da direita, o filtro do ruido diminui o ruido, diminuindo
% tambem o contraste.
% Em suma, WZ = aumenta contraste -> aumenta ruido;
% Wn = diminui ruido -> diminui contraste;
% Logo, o objetivo eh encontrar um filtro combinado que possa aumentar o
% contraste da imagem, mas ao mesmo tempo nao gerar muito ruido.

%% Exercise 5:

% O filtro combinado eh aplicado na imagem original (X) e na imagem com
% ruido (Xn).

eta = 0.7*std(x(:));

Wn = (C + eta^2*eye(144))\C;

Wcombined = WZ*Wn;

w = reshape(Wcombined(78, :), 12, 12);

Z = conv2(X, w, 'same');

Xn = X + eta*randn(size(X));

Zn = conv2(Xn, w, 'same');

figure('Name', 'images')
subplot(2, 2, 1)
imagesc(X);
subplot(2, 2, 2)
imagesc(Z, [-1 1]);
subplot(2, 2, 3)
imagesc(Xn);
subplot(2, 2, 4)
imagesc(Zn, [-1 1]);
colormap('gray')

% A figura mostra a imagem original (X), a imagem convolucionada com o
% filtro (Z), mesma imagem original com ruido (Xn) e a image Xn
% convolucionada com o filtro (Zn). É possivel verificar nessa imagem que o
% filtro eh capaz de recuperar a informacao original de uma imagem mesmo
% com a presenca de ruido.

%% plot the ICA filters
figure('Name', 'ICA filters')
for i = 1:25
    subplot(5, 5, i)
    imagesc(reshape(WI(i, :), 12, 12)); colormap('gray'); axis off; axis square
end

% O filtro separa features independentes da imagem original e
% representa cada uma delas em uma imagem.
% A imagem plota os 25 primeirs filtros ICA. Cada filtro eh uma mascara
% que, ao ser convoluida com a imagem original, extrai uma feature da
% imagem.


%% Exercise 6: plot a histogram of Z and ZICA

% Um histograma das respostas do sinal decorrelacionado e um histograma das
% respostas do sinal após a separacao pelo filtro ICA. O eixo x representa
% as possiveis respostas que os sinais podem assumir, enquanto o eixo y
% representa o numero de ocorrencias com que essas respostas occorem.

grd = -100:1:100; % grid for histogram

Z = WZ*x;
ZICA = WI*x;

pz = hist(Z(:), grd);
pr = hist(ZICA(:), grd);

figure('Name', 'responses')
semilogy(grd, pz, 'k'); hold on
semilogy(grd, pr, 'r'); hold on
legend('decorrelated', 'independent component')

% O filtro descorrelacionado busca aumentar o contraste da imagem,
% diminuindo a variacao de valores de pixels. Assim, ha uma quantidade
% menor de valores diferentes de pixels. Dessa forma, a imagem resultante
% possui uma menor quantidade de informacao sobre a imagem original.
% Enquanto o filtro ICA busca encontrar diferentes independentes, logo, ela
% obtera todas as variacoes de valores de pixels. Assim, nao há uma
% diminuicao da quantidade de valores de pixels, como no filtro
% descorrelacionado.

% A curva decorrelacionada tem uma distribuicao mais concentrada em torno
% de zero, indicando que os valores dos pixels sao mais proximos de zero em
% geral. Ja a curva dos valores independentes eh mais espalhada, indicando
% que os valores dos pixels podem variar mais em relacao a zero. Como o
% objetivo do filtro ICA eh obter uma distribuicao de valores
% independentes, os valores dos pixels após esse filtro ficam de fato
% mais espalhada (tem mais possibilidades de respostas).

%% plot conditional histogram of ZICA

% Plotar um histograma condicional da distribuicao dos dois primeiros
% componentes independentes.

pz1z2 = hist3(ZICA([9, 10], :)', {grd' grd'});
pz2 = sum(pz1z2, 1);

% Densidade de probabilidade condicional, onde cada elemento representa a
% probabilidade de Z1 dado Z2, ou seja, dado um determinado valor de Z2,
% qual eh a probabilidade de que Z1 tenha outro valor.
pz1_z2 = pz1z2./(pz2+1e-5);

figure('Name', 'conditional histogram')
imagesc(grd, grd, log(pz1_z2+0.01))
set(gca, 'Xlim', [-10 10], 'Ylim', [-10 10])
colormap('gray')

% Clores claras = valores mais altos de densidade de probabilidade;
% Cloreas escuras = valores mais baixas de densidade de probabilidade;
% O filtro ICA tem objetivo de obter features independentes. Como nao eh
% possivel obter features totalmente diferentes, ha uma dependencia entre
% uma feature e outra, ou seja, uma feature possui informacoes sobre a
% outra. Nesse grafico, se a feature do eixo x estiver em zero, a
% probabilidade da feature do eixo y ser 0 eh muito alta. Logo, a gente
% consegue conhecer a feature y a partir da feature x, logo, elas estao
% correlacionadas. 


% Esse grafico mostra como a feature 10 depende da feature 9.
% Cada eixo do mapa de calor representa os valores possiveis de cada
% componente. A figura mostra como a distribuicao do componente 9 eh
% afetada pelo valor do componente 10.

