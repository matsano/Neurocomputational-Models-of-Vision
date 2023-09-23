clear all;
close all;

whos -file dataENSTA_Lect_1.mat
load dataENSTA_Lect_1.mat

% binnedOFF = acontece o spike, quando a luminiscencia desce
% binnedON = acontece o spike, quando a luminiscencia sobe
[R T] = size(binnedOFF); % extract number of repetitions and number of time bins per repetition
                         % pega as dimensoes da matriz binnedOFF (dados de
                         % atividade neuronal)
% Numeros de bins de tempo correspondente a 0.5 segundos
integrationTime = 0.5/dt; % set the past-stimulus integration time (0.5sec) as number of time bins

% Define matlab color codes
matCol = [ 0    0.4470    0.7410; 
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];

%% COMPUTE AND VISUALIZE PSTH AND STIMULUS
%
% r = mean( n(t,r) , r )
% psth = r/dt
%

% Janelas de tempo diferentes para treinamento e teste
% timeTr e timeTe sao vetores cujo cada elemento eh o tempo que inicia em
% integrationTime (20)
timeTr = integrationTime:floor(0.66*T); % time bins for training
timeTe = ceil(0.66*T):T; % time bins for testing
% Numero de intervalos de tempo em cada janela
TTr = numel(timeTr); % length of training
TTe = numel(timeTe); % length of testing

% Calcula a média da atividade neural em cada intervalo de tempo
% rTr e rTe sao vetores em que cada elemento eh a media da atividade neural
% em cada intervao de tempo
% for studying OFF cell
rTr = mean(binnedOFF(:,timeTr),1);
rTe = mean(binnedOFF(:,timeTe),1);

% for studying ON cell
%rTr = mean(binnedON(:,timeTr),1);
%rTe = mean(binnedON(:,timeTe),1);

% Divide a media da atividade neural pela duracao do intervalo de tempo
% (intervalo de tempo dt fixo) para ter a media da atividade neural por
% segundo
% compute psth as rate in Hz
psthTr = rTr/dt;
psthTe = rTe/dt;

% Cada barra vertical representa a taxa de disparo medio em cada intervalo
% de tempo
fig=figure;

subplot(2,1,1)
hold on
plot((1:T)*dt,stim,'k','LineWidth',2.0)
xlabel('Time (s)')
ylabel('Luminance')
set(gca,'Fontsize',16);
set(gca,'box','off')

subplot(2,1,2)
hold on
plot(timeTr*dt,psthTr,'LineWidth',2.0,'Color',matCol(1,:))
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
text(0.01,0.95,'Training','Units','normalized','HorizontalAlignment','left','Color',matCol(1,:),'FontSize',18);
text(0.99,0.95,'Testing','Units','normalized','HorizontalAlignment','right','Color',matCol(2,:),'FontSize',18);
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%%  LINEAR MODEL

% f(t) = \sum_tau w(tau) * ( S(t-tau) -mean(S) ) + b

% We seek to minimize 1/2 * \sum_t (f(t) - n(t) ).^2
%
% n(t) -> nTilde(t) = n(t)-mean(n)
% S(t) -> STilde(t) = ( S(t)-mean(S) ) / std(S)
%
% autoCov(tau,tau') = \sum_t STilde(t+tau) STilde(t+tau')
% STA(tau) = \sum_t nTilde(t) * STilde(t-tau)
% w(tau) =  STA * Inv(autoCov) 
% 

%% STIMULUS PRE-PROCESSING

% Intervalo de tempo utilizado (exclui os 19 primeiros intervalos de tempo)
timeRange = integrationTime:T; % exclude the first time bins for training

% Normaliza o estimulo para evitar que diferencas de escala no estimulo
% afetem a modelagem
% STilde
stimTilde = ( stim' - mean(stim) ) / std(stim); % normalize stimulus
fullStim = zeros([T integrationTime]); % for convenience expand stimulus fullStim(t,tau) = stim(t-tau+1)

% Expande o estimulo em uma matriz de tamanho [T integrationTime] em que
% cada linha corresponde a um instante de tempo e cada coluna corresponde a
% uma janela de integracao
for tt=timeRange
    fullStim(tt,:) = stimTilde((tt-integrationTime+1) :tt);
end

%% STIMULUS AUTOCORRELATION

% Vetor da autocorrelacao do estimulo para diferentes atrasos de tempo
stimAutoCorr = fullStim(timeTr,:)' * fullStim(timeTr,:); % compute stimulus autocorrelation

fig=figure;
hold on
% A cada linha da matriz stimAutoCorr, o pico esta em cada tempo (0 a 20)
% A linha diagonal representa a autocorrelacao do estimulo quando nao ha um
% atraso de tempo. As outras linhas representam a autocorrelacao do
% estimulo para diferentes atrasos de tempo.
% A cada linha eh deslocado cada elemento uma unidade pra direita
plot( 0:integrationTime-1,stimAutoCorr(1,:)/TTr,'Linewidth',2.0,'Color',matCol(1,:) )
xlabel('Time (s)')
ylabel('Auto correlation')
ylim([0 1]);
set(gca,'Fontsize',16);
set(gca,'box','off')

% A figura mostra como a autocorrelacao do estimulo visual varia com o
% atraso de tempo.
% Como a autocorrelacao do estimulo diminui, os valores dos estimulos
% observados são cada vez menos dependentes dos valores anteriores.
% À medida que a distância temporal entre esses dois momentos aumenta,
% a correlação entre eles diminui. Isso ocorre porque o estímulo pode mudar
% significativamente ao longo do tempo, de modo que um estímulo anterior
% pode ter pouca relação com o estímulo atual. Além disso, o processo que
% gera o estímulo pode ter alguma aleatoriedade, o que pode tornar o
% estímulo em momentos distintos ainda mais diferentes.
% À medida que o tempo aumenta, um dado se torna independente de outro.

%% STA = Spike-Triggered Average, and LINEAR FILTER
%
% STA(tau) = \sum_t nTilde(t) * STilde(t-tau)
% w(tau) =  STA * Inv(autoCov) 
% b = mean( n(t) )
%
% But we will first ignore off diagonal elements in autoCorrelation
%
%

% n(t) = media da atividade neural (spikes counts)
b = mean(rTr); % set the constant of the model
rTrTilde = rTr - b; % remove the mean from the response

% Spike-Triggered Average
% STA representa o estimulo medio que o neuronio experimentou na janela de
% tempo que precede o spike, logo, STA eh a medida do estimulo que leva a
% um spike.
STA = rTrTilde * fullStim(timeTr,:); % compute STA

% Consideramos o autocorr maximo (que esta na diagonal principal)
% wLin eh um filtro linear e representa os pesos que serao multiplicados
% com o estimulo para estimar a resposta do neuronio.
wLin = STA * inv( diag(diag( stimAutoCorr ) ) ); % for the moment we ignore the offdiagonal autocorrelation

% O grafico mostra o filtro linear em funcao do tempo.
fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'Linewidth',2.0 ,'Color',matCol(3,:))
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')

% O grafico mostra como o estímulo é ponderado em diferentes tempos
% anteriores à resposta do neuronio. Esse filtro wLin multiplica o estimulo
%  para prever a resposta. Nesse caso, por se tratar de um
% "binnedOFF", o filtro deve gerar um spike quando eh borda de descida.
% Dessa forma, os pesos do filtro são positivos quando o estimulo eh alto,
% gerando uma multiplicacao muito positiva. Os pesos do filtro são
% negativos quando o estimulo eh mais baixo, assim vai diminuir um pouco o
% resultado da multiplicacao, porém vai continuar um resultado da
% multiplicacao positiva, gerando o spike.
% Se fosse uma borda de subida, os coeficientes positivos do filtro iriam
% ser multiplicados pela parte mais baixa do estimulo, enquanto os coeficientes
% negativos multiplicam pela parte mais alta do estimulo, gerando um
% resultado negativo, não gerando um spike.

%% LINEAR PREDICTION

% f(t) = \sum_tau w(tau) * ( S(t-tau) -mean(S) ) + b

% Regressao linear para prever a taxa de disparo neural na janela de teste
% com base no estimulo apresetado na janela de treinamento

% Eh usado o filtro linear obtido (wLin) para prever a taxa de disparo
% (psth) da celula para um conjunto de estimulos teste.
% fLin = eh a predicao linear da taxa de disparo para cada intervalo de
% tempo
% fullStim = matriz de estimulos de teste
% b = media dos dados de resposta de treinamento
fLin = wLin * fullStim(timeTe,:)' + b; % compute linear prediction

% Para verificar a performance do modelo de regressao linear, usa a
% correlacao de Pearson entre a taxa de disparo real (psthTe) e a taxa de
% disparo predita (fLin)
['perf lin. model = ' num2str( corr(psthTe', fLin'))]

% Printa o grafico apenas da fase de teste
fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fLin/dt,'LineWidth',1.0,'Color',matCol(3,:))
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

% Como pode ser analisada, a previsao nao eh mt boa, pois ela apresenta os
% picos que acontecem, porém com uma baixa precisao. Ademais, a previsao
% apresenta uma taxa de disparo em Hz negativa, o que é um erro.

%% WHAT IS HAPPENING ??

% Do a scatter plot of psth against prediction

fig=figure;
hold on
plot(psthTe,fLin/dt,'.','MarkerSize',12)
plot([-100 100],[-100 100],'--k')
xlabel('PSTH')
ylabel('PREDICTION')
set(gca,'Fontsize',16);
set(gca,'box','off')
% Do you understand this plot ?
% Essa parte do código faz um gráfico de dispersão (scatter plot) da taxa de disparo observada (psth) versus
% a taxa de disparo prevista (fLin/dt).
% A linha pontilhada em preto representa uma reta diagonal onde os valores observados e previstos seriam iguais.
% Essa plotagem é útil para avaliar o ajuste do modelo linear.
% Se os pontos estiverem próximos da linha diagonal, isso indica um bom ajuste, ou seja,
% o modelo linear é capaz de prever a taxa de disparo observada com precisão.
% Por outro lado, se os pontos estiverem muito distantes da linha diagonal,
% isso indica um ajuste ruim, o que sugere que o modelo linear não é capaz de prever
% com precisão a taxa de disparo observada.

% Esse grafico mostra o quao disperso estao os pontos da previsao e os
% pontos observados, o que confirma uma previsao pouco performante.

%% ReLU TRUNCATION
%
% ReLU(x) = max(x,0)
%
% fReLU = ReLU( fLin )
%

% A funcao ReLU eh uma funcao nao linear que eh usada para introduzir nao
% linearidade em modelos lineares. A saida da funcao ReLU eh o maximo entre
% 0 e o resultado da operacao linear.
fReLU = max(fLin,0);
['perf lin. Model with ReLU = ' num2str(  corr(psthTe', fReLU'))]

fig=figure;
hold on
plot(timeTe*dt,fReLU/dt,'LineWidth',1.0,'Color',matCol(3,:))
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

% Essa funcao eh aplicada para eliminar a taxa de disparo negativa, como
% pode ser visto no grafico plotado. E podemos ver que a performance
% printada no terminal melhorou.

%% INCLUDING AUTOCOVARIANCE
%
% STA(tau) = \sum_t rTilde(t) * xTilde(t-tau)
% w(tau) =  STA * Inv(autoCov)
% b = mean( r(t) )
%
% And now with the full autoCorrelation
%

% Agora repete os calculos considerando toda a matriz de autocovariancia,
% isto eh, sao considerados as autocorrelacoes entre diferentes tempos de
% estiulos.
wLinAC = STA * inv( stimAutoCorr ); % Using the full autocovariance matrix

fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0,'Color',matCol(3,:)) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0,'Color',matCol(4,:)) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')
% Do you understand this plot ?
% No modelo anterior (considerando apenas a diagonal principal da
% autocorrelacao), o estimulo tinha que ser antes de t=-0.35s para aumentar
% a probabilidade do neuronio disparar. Já com esse novo metodo
% (considerando toda a matriz de autocorrelacao), eh possivel realizar o
% impulso em varios momentos t diferentes para aumentar a probabilidade de
% o neuronio disparar.
% Ademais, ao considerar toda a matriz de autocorrelacao, são consideradas
% todas as relacoes entre os estimulos em diferentes atrasos temporais.
% Logo, a curva wLinAC acaba sendo mais precisa.

% Prever a taxa de disparo considerando agora toda a matriz de
% autocorrelacao (a autocorrelacao entre diferentes tempos de estimulos). A
% funcao ReLU ja eh aplicada.
fReLUAC = max( wLinAC * fullStim(timeTe,:)' + b , 0 ); % compute prediction

['perf Lin. model complete with ReLU = ' num2str( corr(psthTe', fReLUAC') )]

fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fReLU/dt,'LineWidth',1.0,'Color',matCol(3,:))
plot(timeTe*dt,fReLUAC/dt,'LineWidth',1.0,'Color',matCol(4,:))
xlim([10 15]);
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

% O grafico e o valor da performance printado mostram que considerar a
% matriz de autocorrelacao inteira melhora a precisao e a performance da
% previsao.

%% LN MODEL FITTING
%
% f(x) = exp( \sum_tau w(tau) * x(t-tau) +b)
% log p(n|x) = log Poiss(n|x) = n * log(f(x)) - f(x) 
%

% Modelo LN (Linear-Nonlinear): generalizacao do modelo STA e inclui uma
% funcao nao linear (neste caso, a exponencial).
% O objetivo eh encontrar os melhores parametros para a funcao exponencial
% que maximizem a probabilidade dos dados observados.
% Maximizar a log-verossimilhanca equivale a encontrar os parametros que
% tornam os dados observados mais provaveis. Logos, usaremos
% log-verossimilhanca.

% Os parametros estimados sao inicializados com valores aleatorios e muito
% pequenos.
% initiliase parameters
wLn = 1e-9*randn(1, integrationTime);
bLn = 0;

% Para ajustar os parametros desejados (wLn e bLn), eh utilizado o metodo
% de gradiente descendente, que ajusta os parametros em pequenos
% incrementos (eta) para maximizar a probabilidade dos dados.

% Os parametros do modelo sao inicializados (Nit o numero maximo de
% iteracoes e eta a taxa de aprendizagem)
% parameters for gradient descent
Nit = 500;                  % number of iterations
eta = 1e-1;                 % step size

Ltr = zeros(1, Nit);          % log likelihood training
Lte = zeros(1, Nit);          % log likelihood testing
for i = 1:Nit
    
    % Calculo da taxa de disparo predita
    % firing rate prediction
    % f(x) = exp( \sum_tau w(tau) * x(t-tau) +b)
    fLNtr = exp(  wLn * fullStim(timeTr,:)' + bLn );
    fLNte = exp(  wLn * fullStim(timeTe,:)' + bLn );
    
    % Calculo da log-verossimilhanca (log da funcao de densidade de
    % probabilidade p dos dados observados n dado o conjunto de parametros
    % representado pela taxa de disparo predita f(x)).
    % log p(n|x) = log Poiss(n|x) = n * log(f(x)) - f(x) 
    % log-likelihood training
    ll = log(fLNtr) .* rTr - fLNtr;
    Ltr(i) = mean(ll);

    % log-likelihood testing
    ll = log(fLNte) .* rTe - fLNte;
    Lte(i) = mean(ll);

    % Calculo do gradiente da log-verossimilhanca
    % Se o gradiente for positivo, o valor do parametro deve ser aumentado
    % para a verossimilhanca aumentar. Se o gradiente for negativo, o valor
    % do parametro deve diminuir para a verossimilhanca aumentar.
    % Como queremos maximizar a log-verossimilhanca, calculamos o
    % gradiente.
    % derivative of log likelihood
    dL_w = ( rTr - fLNtr ) * fullStim(timeTr,:) /TTr;
    dL_b = sum(rTr-fLNtr)/TTr;

    % Atualizacao dos parametros usando o gradiente descendente.
    % update parameters 
     wLn = wLn + eta * dL_w; 
     bLn = bLn + eta * dL_b;

end

% Grafico da log-verossimilhanca ao longo do tempo (iteracoes)
fig=figure;
hold on
plot(1:Nit,Ltr,'--','LineWidth',2.0,'Color',matCol(5,:))
plot(1:Nit,Lte,'LineWidth',2.0,'Color',matCol(5,:))
%xlim([10 15])
xlabel('Epoch')
ylabel('Log Likelihood')
set(gca,'Fontsize',16);
set(gca,'box','off')

% Como o objetivo do metodo eh maximizar a log-verossimilhanca, ou seja,
% encontrar os valores dos parametros que produzem a maior probabilidade de
% observar os dados de treinamento e teste, a log-verossimilhanca deve ser
% aumentada À medida as iteracoes avancam, a log-verossimilhanca aumenta,
% indicando que o modelo esta melhorando.

% Idealmente, gostaríamos de maximizar a log-verossimilhança nos dados de teste,
% pois estes são dados que não foram usados para treinar o modelo e, portanto,
% fornecem uma medida mais realista do desempenho do modelo em dados não vistos.
% Se a log-verossimilhança nos dados de teste começar a diminuir em relação
% à log-verossimilhança nos dados de treinamento, isso pode ser um sinal de
% que o modelo está sofrendo overfitting.

%% INFERRED FILTER, PREDICTION AND PERFORMANCE

% Plota wLin para os 3 metodos apresentados (regressao linear considerando
% diagonal principal, considerando matriz completa de autocorr e pelo
% modelo LN).
fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0,'Color',matCol(3,:)) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0,'Color',matCol(4,:)) % comment
plot((1-integrationTime:0)*dt,wLn,'LineWidth',2.0,'Color',matCol(5,:)) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')
% How does it look?
% wLin e wLinAC sao semelhantes, o que indica que a matriz de
% autocovariancia dos estimulos nao contribuiu muito para melhorar a
% qualidade da predicao. A curva wLn eh diferente das outras, pois utiliza
% informacoes nao lineares sobre a taxa de disparo das celulas para estimar
% o filtro.

% Quanto mais proximo de zero o filtro, menor eh a influencia daquele
% momento passado do estimulo na taxa de disparo atual das celulas. Valores
% positivos e negativos estao associados a um aumento ou diminuicao da taxa
% de disparo das celulas.

% Avalia a performance deste modelo
% A performance diminui
fLNte = exp(  wLn * fullStim(timeTe,:)' + bLn );

['perf LN model = ' num2str( corr(psthTe', fLNte' ))]

% Plota as previsoes para diferentes modelos e a taxa de impulso observada.
fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fReLUAC/dt,'LineWidth',1.0,'Color',matCol(4,:))
plot(timeTe*dt,fLNte/dt,'LineWidth',1.0,'Color',matCol(5,:))
xlim([10 15])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

%% SMOOTHNESS REGULARIZATION

% f(t) = \sum_tau w(tau) * ( x(t-tau) -mean(x) ) + b
% We seek to minimize 1/2 * \sum_t (r(t) - f(t) ).^2 + lambda/2 * w * Laplacian * w
%
% w * Laplacian * w = sum_tau (w(tau) - w(tau+1)).^2
%

% Regularizacao para evitar uma variacao muito grande do estimulo e melhorar a generalizacao do
% modelo. Penalizar solucoes em que a funcao estimada varia muito
% rapidamente em dois instantes, forcando-a a ter uma forma mais suave e continua.

% let's define the laplacian matrix
% Matriz laplaciana eh fixa.
lapl = 4*eye(integrationTime);
lapl = lapl - diag( ones([integrationTime-1 1]),1);
lapl = lapl - diag( ones([integrationTime-1 1]),-1);
lapl(1,1) = 2;
lapl(end,end) = 2;

% O comando imagesc(lapl) é usado para visualizar a matriz Laplaciana,
% onde as entradas maiores são representadas por cores mais claras e as
% entradas menores são representadas por cores mais escuras.
fig=figure;
imagesc(lapl);
colorbar;

%% LN MODEL FITTING WITH SMOOTHNESS REGULARIZATION

% Modelo LN com a regularizacao de suavidade. A regularizacao eh adicionada
% ao modelo para evitar grande variacao e suavizar o resultado.
% Modelo LN: encontrar o valor de w e b que minimiza o desvio entre as
% previsoes do modelo e os dados reais;
% Regularizacao de suavidade: evitar o modelo de superestimar.

% Forca de regularizacao eh definida por lambda.
% set regularization strenght value
lambda = 0.015; % this should be optimized over a validation set

% initiliase parameters
wLnReg = 1e-9*randn(1, integrationTime);
bLnReg = 0;

% parameters for gradient descent
Nit = 500;                  % number of iterations
eta = 1e-1;                 % step size

LtrReg = zeros(1, Nit);          % log likelihood training
LteReg = zeros(1, Nit);          % log likelihood testing
for i = 1:Nit
    
    % firing rate prediction
    fLNtrReg = exp(  wLnReg * fullStim(timeTr,:)' + bLnReg );
    fLNteReg = exp(  wLnReg * fullStim(timeTe,:)' + bLnReg );
    
    % log-likelihood training
    % A diferenca entre a predicao do modelo e as taxas de disparo
    % observada eh levada em consideracao, logo, a regularizacao de
    % suavidade eh adicionada.
    ll = log(fLNtrReg) .* rTr - fLNtrReg ;
    LtrReg(i) = mean(ll)  - 0.5 * lambda * wLnReg * lapl * wLnReg';

    % log-likelihood testing
    ll = log(fLNteReg) .* rTe - fLNteReg;
    LteReg(i) = mean(ll);

    % derivative of log likelihood
    dL_w = ( rTr - fLNtrReg ) * fullStim(timeTr,:) /TTr;
    dL_b = sum(rTr-fLNtrReg)/TTr;

    % update parameters 
     wLnReg = wLnReg + eta * dL_w - lambda * wLnReg * lapl; 
     bLnReg = bLnReg + eta * dL_b;

end

% Graficos da log-verossimilhanca ao longo do tempo (iteracoes) sem
% regularizacao e com regularizacao.
fig=figure;
hold on
plot(1:Nit,Ltr,'--','LineWidth',2.0,'Color',matCol(5,:))
plot(1:Nit,Lte,'LineWidth',2.0,'Color',matCol(5,:))
plot(1:Nit,LtrReg,'--','LineWidth',2.0,'Color',matCol(6,:))
plot(1:Nit,LteReg,'LineWidth',2.0,'Color',matCol(6,:))
%xlim([10 15])
xlabel('Epoch')
ylabel('Log Likelihood')
set(gca,'Fontsize',16);
set(gca,'box','off')

% A regularizacao de suavidade tem um impacto positivo no desempenho do
% modelo, pois a log-verossimilhanca para o conjunto de teste aumenta com a
% regularizacao. Ademais, a diferença entre as curvas de treinamento e
% teste eh menor com a regularizacao, o que indica que a curva de teste
% nao diminui e portanto, o modelo nao sofre mais overfitting.

%% INFERRED FILTER, PREDICTION AND PERFORMANCE

% Plota wLin para os 4 metodos apresentados (regressao linear considerando
% diagonal principal, considerando matriz completa de autocorr, pelo
% modelo LN sem regularizacao de suavidade e pelo modelo LN com regularizacao).
fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0,'Color',matCol(3,:)) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0,'Color',matCol(4,:)) % comment
plot((1-integrationTime:0)*dt,wLn,'LineWidth',2.0,'Color',matCol(5,:)) % comment
plot((1-integrationTime:0)*dt,wLnReg,'LineWidth',2.0,'Color',matCol(6,:)) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')
% How does it look?
% Com a regularizacao, os pesos tendem a permanecer menores em magnitude, o
% que é esperado, ja que a regularizacao visa evitar overfitting. Ademais,
% a evolucao dos pesos com a regularizacao eh mais suave, uma vez que a
% regularizacao visa suavizar a trajetoria do aprendizado.

% Calculo da performance. Apresenta uma performance melhor!
['perf LN model with Reg = ' num2str( corr(psthTe', fLNteReg' ))]

% Taxa de disparo real dos dados de teste e as 4 previsoes.
fig=figure;
hold on
plot(timeTe*dt,psthTe,'LineWidth',2.0,'Color',matCol(2,:))
plot(timeTe*dt,fReLUAC/dt,'LineWidth',1.0,'Color',matCol(4,:))
plot(timeTe*dt,fLNte/dt,'LineWidth',1.0,'Color',matCol(5,:))
plot(timeTe*dt,fLNteReg/dt,'LineWidth',1.0,'Color',matCol(6,:))
xlim([10 15])
ylim([0 200])
xlabel('Time (s)')
ylabel('Spiking Rate (Hz)')
set(gca,'Fontsize',16);
set(gca,'box','off')

% Assim como pode ser visto, a previsao com modelo nao linear com
% regularizacao apresenta a melho performance e se aproxima mais da taxa de
% disparo real.

%% LAMBDA OPTIMIZATION

% Encontrar a melhor configuracao de regularizacao.

% Gerar 30 lambdas aleatorios que variam entre 10^-2 e 10^4.
lambdaRange = logspace(-2,4,30);
nLambda = numel(lambdaRange);
perfLambda = zeros([nLambda 1]);

% Para cada valor de lambda, calcula os parametros w e b, e calcula a
% correlacao entre a resposta prevista e a resposta real. Armazena o
% resultado em um vetor perfLambda.
for ll = 1:nLambda
    lambda=lambdaRange(ll);
    wLinReg = STA * inv( stimAutoCorr + lambda * lapl );
    fLinReg = wLinReg * fullStim(timeTe,:)' + b; % comment

    perfLambda(ll) =  corr(psthTe', fLinReg');
end

% Plota a performance (correlacao) em funcao de lambda.
% Achar o lambda cuja performance seja a maior.
fig=figure;
hold on
plot(lambdaRange,perfLambda,'LineWidth',2.0)
set(gca,'XScale','log')
xlabel('Regularization strenght')
ylabel('Performance')
set(gca,'Fontsize',16);
set(gca,'box','off')

% Uma forca de regularizacao (lambda) moderada eh benefica para melhorar o
% desempenho do modelo, mas uma forca mt alta pode levar a overfitting e
% uma piora no desempenho em dados de teste.

%% COMPUTE OPTIMAL W and ADD ReLU

% Encontra o valor de lambda que resulta no melhor desempenho.
[~,llBest] = max(perfLambda);
lambda = lambdaRange(llBest);

% Encontra o filtro w e a previsao com o melhor lambda.
wLinReg = STA * inv( stimAutoCorr + lambda * lapl );
fLinReg = wLinReg * fullStim(timeTe,:)' + b; % comment

% Obtem o desempenho do modelo com ou sem a funcao ReLu.
perfLinReg = corr(psthTe', fLinReg')
perfReLUReg = corr(psthTe', max(fLinReg,0)')

% Plota wLin para diferentes metodos.
fig=figure;
hold on
plot((1-integrationTime:0)*dt,wLin,'LineWidth',2.0) % comment
plot((1-integrationTime:0)*dt,wLinAC,'LineWidth',2.0) % comment
plot((1-integrationTime:0)*dt,wLinReg,'LineWidth',2.0) % comment
plot( [-dt*integrationTime 0],[0 0],'--k')
xlabel('Past time (s)')
ylabel('wLin')
set(gca,'Fontsize',16);
set(gca,'box','off')

% Assim como visto anteriormente, com a regularizacao, os pesos tendem a
% permanecer menores em magnitude, o que é esperado, ja que a regularizacao
% visa evitar overfitting. Ademais, a evolucao dos pesos com a regularizacao
% eh mais suave, uma vez que a regularizacao visa suavizar a
% trajetoria do aprendizado.

% A performance com a ReLU eh melhor do que sem a ReLU. Ademais, usando
% esse ultimo metodo de identificar o lambda, nos obtemos uma performance
% melhor do que antes, chutando um lambda qualquer.
