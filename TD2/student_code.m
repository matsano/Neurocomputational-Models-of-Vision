clear all
close all

%% creat fake data

% range of stimulus coherences
% Coerencia do estimulo = eh uma medida que descreve o grau de
% sincronizacao ou correlacao entre dois sinais. Uma alta coerencia indica
% que as atividades neurais estão sincronizadas, enquanto uma baixa
% coerencia indica que as atividades neurais estao ocorrendo de forma
% independente.
coherence = [0.025 0.05 0.1 0.2 0.3 0.5 0.7];
% Numero de estimulos
nstim = numel(coherence);                       % number of stimuli

% Contagem media de picos de fundo na ausencia de qualquer estimulo
mean_background = 10;   % background mean spike count
% lamdba = contagem media de picos para cada nivel de coerencia do estimulo
lambda = mean_background+30*coherence;                       % mean spike count 
% Numero de tentativas
ntr = 1e3;                                      % number of trials
% Contagem de picos para a media de fundo para cada tentativa usando
% distribuicao de Poisson (em uma situacao que nao ha estimulo).
r0 = poissrnd(repmat(mean_background, ntr, 1));          % generate spikes (0% coherence)
% Contagem de picos para cada nivel de coerencia para cada tentativa,
% usando a distribuicao Poisson.
% Cada linha representa as tentativas e cada coluna representa cada
% coerencia.
r = poissrnd(repmat(lambda, ntr, 1));       % generate spikes

%% plot histograms

% x = diferentes contagens de spikes para o histograma.
x = 0:50;                  % different spike counts for histogram
edges =  [x-0.5,x(end)+1]; % bin edges for histogram

% Histograma de contagem de spikes para o nivel de coerencia zero.
n0 = histcounts(r0, edges);            % histogram of spike counts (0% coherence)

% Para cada nivel de coerencia, eh plotado um novo histograma.
figure('Name', 'firing rate histograms')
for i = 1:nstim
    n = histcounts(r(:, i), edges);   % histogram of spike counts
   
    subplot(nstim, 1, i) 
    bar(x, [n0; n])
    ylabel('trials')
    title(sprintf('coherence = %.1f %%', coherence(i)*100));
    
end
xlabel('spike count')

% Os histogramas mostram no eixo x o numero de spikes que apareceram, enquanto no eixo y
% mostra o numero de contagens (vezes) que cada numero de spikes foi
% observado. O histograma azul eh para um nivel de coerencia zero, enquanto
% o histograma laranja eh varia de acordo com a coerencia.
% A medida que a coerencia aumenta, a distribuicao de contagem de spikes
% muda, com um aumento geral no numero de spikes (uma resposta neural mais forte ao
% estimulo) e um distribuicao mais ampla (um aumento da coerencia do
% estimulo esta levando a uma maior variabilidade nas respostas neurais).

%% ROC curves

% A curva ROC eh um grafico que permite avaliar o desempenho de um
% classificador binario (classifica um item em duas categorias). A curva
% ROC mostra a relacao entre a taxa de verdadeiros positivos e a taxa de
% falsos positivos para diferentes valores de thresholds.

% Quando o classificador eh perfeito, a curva ROC eh um degrau que sobre de
% (0,0) ateh (0,1) e, depois, permanece com beta 1. Isso representa uma
% taxa de verdadeiros positivos de 1 para todos os valores de taxa de
% falsos positivos, logo, um desempenho perfeito.

% Quando o classificador eh aleatório, a curva ROC eh uma diagonal que vai
% de (0,0) ateh (1,1). Qualquer curva ROC acima dessa linha representa um
% desempenho melhor do que o aleatório, isto eh, o classificador possui
% uma taxa de verdadeiro positivo maior do que de falso positivo.

% z = limiares de decisao (thresholds)
z = 50:-1:0;                            % thresholds
nz = numel(z);                          % number of thresholds

% HIT = proporcao de estimulos que foram considerado de fato estimulos
% FALSE ALARM = proporcao de ruidos que sao considerados estimulos
alpha = zeros(nz, 1);                   % false alarm rate
beta = zeros(nz, nstim);                % hit rate

% loop over thresholds
% Para cada thresholds, calcula a taxa de falsos positivos (alfa) e a taxa de
% verdadeiros positivos (beta).
% Se a atividade neural excede o valor de corte, é considerada uma
% resposta. Caso contrario, nao eh uma resposta.
for i = 1:nz
    alpha(i) = mean(r0>z(i));             % false alarm rate
    beta(i, :) = mean(r >= z(i));         % hit rate
end

% plot ROC curve
figure('Name', 'ROC curve')
plot(alpha, beta, 'o-', 'Linewidth', 2); hold on
plot([0 1], [0 1], 'k--'); hold off
xlabel('alpha')
ylabel('beta')

% A medida que a coerencia do estimulo aumenta, a curva ROC se desloca para
% cima e para a esquerda, indicando um aumento na capacidade de discriminar
% o sinal do ruido, ou seja, o desempenho melhora. A coerencia do estimulo
% esta relacionada à diferenca entre as distribuicoes de respostas corretas
% e incorretas dos neuronios. Quanto maior a coerencia, maior esta
% diferenca e, portanto, mais facil eh para o classificador distinguir
% entre as respostas corretas e incorretas dos neuronios.

% O threshold varia de 50 à 0. À medida que diminuimos o threshold, mais
% estimulos comecam a ser considerados estimulos, por isso que a taxa de
% acerto (beta) aumenta, chegando em um ponto que todos os estimulos sao
% considerados estimulos e nenhum ruido eh considerado estimulo (localizado
% no ponto (1,0)). Porem, à medida que o threshold diminui, o ruido comeca
% a ser considerado como estimulo e, assim, a taxa de falso positivo (alfa)
% comeca a aumentar. Quando o threshold eh zero, todos os estimulos sao
% considerados estimulos e todos os ruidos tambem sao considerados
% estimulos (ponto (1,1)).

%% area under curve and performance in 2 alternative forced choice experiment 

% Comparacao da performance de um neuronio em discriminar entre o estimulo
% e o ruido de fundo. 

% compute area under curve
% Eh calculada a diferenca entre os valores de alfa em cada par de pontos
% consecutivos. Em seguida, eh calculada a multiplicacao dessa diferenca
% com a diferenca entre os valores de beta. Somando os produtos das
% multiplicacoes, obtem-se a area sob a curva ROC (AUC).
% Como visto nos slides, a curva sob a curva ROC (integral de beta em
% funcao de alpha) representa a probabilidade correta.
dalpha = alpha(2:end)-alpha(1:end-1);
AUC = dalpha'*beta(1:end-1,:) ;

% TWO ALTERNATIVE FORCED CHOICE (2AFC)
% Comparando dois estimulos em ordem aleatoria, estima a probabilidade
% correta diretamente comparando esses dois estimulos.
% Em um experimento no qual deve ser escolhida entre dois estimulos, a taxa
% de acerto eh a proporcao de respostas corretas em relacao ao numero total
% de escolhas (quantas vezes o neuronio identifica corretamente quais das
% duas opcoes eh de fato um estimulo).
% A taxa de acerto aumenta à medida que a coerencia do estimulo
% aumenta, ja que estimulos mais fortes sao mais faceis de discriminar.
% compute error in 2AFC
p2AFC = mean(r >= r0);

figure('Name', 'neuronal')
semilogx(100*coherence, AUC, '-o'); hold on
semilogx(100*coherence, p2AFC, 'r-o'); hold on
xlabel('log coherence')
ylabel('area under curve')
leg = legend('area under curve', 'probability correct');
set(leg, 'Location', 'SouthEast', 'Box', 'off', 'Fontsize', 12)

% Como pode ser visto pelo grafico, a probabilidade correta calculada
% diretamente comparando dois estimulos eh proxima da probabilidade correta
% obtida por meio da area da curva. Porém, a AUC eh uma medida mais
% sensivel do que a taxa de acerto em 2AFC. Isso ocorre, pois a AUC leva em consideracao
% tanto a taxa de acerto quanto a taxa de falsos alarmes, enquanto a taxa
% de acerto em 2AFC se concentra apenas na discriminacao correta.

%% compute entropy for binary stimulus

% Probabilidade de um neuronio disparar ou nao em resposta a um estimulo.
p = linspace(0, 1, 100);                 % probability that x = 1

% Entropia eh relacionada à incerteza/aleatoriedade. Se a entropia eh grande, ha mta
% incerteza.
H =  -p.*log2(p) - (1-p).*log2(1-p);       % entropy

figure('Name', 'entropy of a binary stimulus')
plot(p, H)
xlabel('p') 
ylabel('H')

% O grafico mostra que a entropia eh maxima (quando a distribuicao eh mais
% aleatoria) quando a probabilidade eh 0.5, ou seja, quando os dois
% resultados binarios sao igualmete provaveis.

%% compute mutual information between stimulus and response

% p(x) = 1/nstim

% O logaritmo da probabilidade a priori de uma determinada coerencia eh calculado. O
% logaritmo eh igual para cada uma das coerencias uma vez que a probabilidade
% de todas as coerencias eh a mesma.
% log p(x)
logpx = -repmat(log(nstim), 1, nstim);

% maximum firing rate
rmax = 100;

% Logaritmo da probabilidade condicional de que um determinado
% numero de spikes(r) seja observado dado uma coerencia (x).
% log p(r|x)
logpr_x = zeros(rmax+1, numel(coherence));
for k = 0:rmax
    logpr_x(k+1, :) = k*log(lambda) - lambda - gammaln(1+k);
end

% Probabilidade conjunta de observar uma coerencia e uma resposta a esse
% estimulo, multiplicando a probabilidade condicional e a priori para cada
% combinacao possivel de r (numero de spikes) e x (coerencia)
% compute p(r, x)
prx = exp(logpr_x+logpx);

% Probabilidade marginal da resposta, somando as probabilidades conjuntas
% sobre todas as coerencias possiveis.
% compute p(r)
pr = sum(prx, 2);

% Entropia da resposta (como incerto eh a resposta)
% compute HR
HR = - pr(:)'*log(pr(:));

% Entropia da resposta condicional ao estimulo (como incerto eh a resposta
% depois de calcular o estimulo)
% compute HR_X
HR_X = - prx(:)'*logpr_x(:);

% Informacao mutua = quantidade de informacao que uma variavel contém sobre
% outra variavel. Neste caso, é a quantidade de informacao em nats que o
% neuronio codifica a partir de uma coerencia. Em outras palavras, eh a
% quantidade de informacao que a resposta (spikes) tem sobre a coerencia
% do estimulo.
% Quando a informacao mutua eh alta, a resposta da celula contem muita
% informacao sobre a coerencia (sabe mais sobre qual eh a coerencia do estimulo).
% Quando eh baixa, a resposta contem pouca informacao sobre o estimulo.
% Eh a reducao da incertitude da resposta quando o estimulo eh calculado.
% compute Info I(R;X)
Inf = HR - HR_X;

fprintf('\n neuron encodes %.3f nats\n', Inf)

%% vary stimulus distribution

% A distribuicao do estimulo eh variada pela alteracao do parametro gain
% que multiplica com a coerencia.
gain = [5 30 80 200 300 450 600];

% Para cada valor de gain, o valor da informacao mutua eh calculado igual
% no item acima.
% Ao aumentar o ganho, a gente aumenta a quantidade espera de spikes, logo,
% a gente aumenta a quantidade maxima de spikes que o neuronio pode gerar.
Inf = zeros(numel(gain), 1);
for i = 1:numel(gain)
    
    lambda_new = mean_background+gain(i)*coherence;                       % mean spike count 


    % maximum firing rate
    rmax = max(2*(gain(i)+10), 100);

    % log p(r|x)
    logpr_x = zeros(rmax+1, numel(coherence));
    for k = 0:rmax
        logpr_x(k+1, :) = k*log(lambda_new) - lambda_new - gammaln(1+k);
    end

    % compute p(r, x)
    prx = exp(logpr_x+logpx);

    % compute p(r)
    pr = sum(prx, 2);

    % compute HR
    HR = - pr(:)'*log(pr(:));
    HX = - exp(logpx(:))'*logpx(:);

    % compute HR_X
    HR_X = - prx(:)'*logpr_x(:);

    % compute Info
    Inf(i) = HR - HR_X;
   
end

% O eixo x representa o maximo numero de spikes que o neuronio pode gerar
% em resposta ao estimulo variado. O eixo y representa a informacao mutua
% entre um estimulo variado e a resposta do neuronio.
% A linha representa a entropia da distribuição de probabilidade do estimulo variado (H(x)),
% que eh o limite superior para a informacao mutua. A linha eh constante,
% pois o estimulo eh equiprovavel (a probabilidade de cada valor do
% estimulo eh igual).
% A curva eh a informacao mutua ( H(X) - H(X|R) ). A informacao mutua
% aumenta à medida que o numero maximo de spikes que o neuronio pode gerar
% aumenta, uma vez que o neuronio pode gerar mais spikes e, assim, pode extrair
% mais informacao. Porém, quando um numero maximo de spikes eh atingido, o neuronio
% nao consegue codificar mais informacoes uteis sobre o estimulo variado,
% logo, a informacao mutua estabiliza.
figure('Name', 'Inf versus sigma(x)')
plot(gain+10, Inf, '-o'); hold on
plot([0, max(gain+10)], HX*[1 1]); hold off
xlabel('maximum spike count')
ylabel('Information (nats)')

%% vary background firing rate


