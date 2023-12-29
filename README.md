--MUDAR TODO O READ ME

# Weekly Eletro Plaza Sales Forecasting

<div align="center">
<img src="img/img_capa.png" />
</div>

# Introdução

Este é um projeto end-to-end de Data Science de um modelo de regressão adaptada para séries temporais. E foi proposto como problema de negócio para o Hackday 7 da Comunidade DS.

O objetivo principal desse projeto é fazer a previsão das vendas semanais da última semana do ano da empresa Eletro Plaza, e como as últimas semanas do ano contém eventos como Black Friday e Natal, a métrica usada para rankear os melhores times na competição foi o RMSE, onde as 3 equipes com os menos valores do erro se classificavam para a final.

Este repositório contém a solução para o problema do Kaggle (criado especificamente para o Hackday 7) que o time utilizou para chegar no TOP 3 da competição e se classificar para a final: [https://www.kaggle.com/competitions/test-hackday-7/overview](https://www.kaggle.com/competitions/test-hackday-7/overview)

Esse projeto faz parte da "Comunidade DS", que é um ambiente de estudo que promove o aprendizado, execução, e discussão de projetos de Data Science.

### Plano de Desenvolvimento do Projeto de Data Science

Esse projeto foi desenvolvido seguindo o método CRISP-DS(Cross-Industry Standard Process - Data Science). Essa é uma metodologia capaz de transformar os dados da empresa em conhecimento e informações que auxiliam na tomada de decisão. A metodologia CRISP-DM define o ciclo de vida do projeto, dividindo-as nas seguintes etapas:

- Entendimento do Problema de Negócio
- Coleção dos Dados
- Limpeza de Dados
- Análise Exploratória dos Dados
- Preparação dos Dados
- Modelos de Machine Learning, Cross-Validation e Fine-Tuning.
- Avaliação dos Resultados do Modelo e Tradução para Negócio.
- Modelo em Produção

![crisp!](img/crisp.png)

Nesse projeto em específico, como era voltado para a competição, o plano do time era fazer ciclos rápidos e gerar submissões para pontuar cada vez melhor na competição.

Dessa forma, os capítulos de Avaliação dos Resultados do Modelo e Tradução para Negócio e Modelo em Produção não foram realizados durante a competição, pois o intuito era treinar o modelo e gerar um RMSE menor que o baseline proposto.

### Planejamento

- [1. Descrição e Problema de Negócio](#1-descrição-e-problema-de-negócio)
- [2. Base de Dados e Premissas de Negócio](#2-base-de-dados-e-premissas-de-negócio)
- [3. Estratégia de Solução](#3-estratégia-de-solução)
- [4. Exploration Data Analysis](#4-exploration-data-analysis)
- [5. Seleção do Modelo de Machine Learning](#5-seleção-do-modelo-de-machine-learning)
- [6. Performance do Modelo](#6-performance-do-modelo)
- [7. Pontuação Final](#7-pontuação-final)
- [8. Conclusão](#8-conclusão)
- [9. Aprendizados e Trabalhos Futuros](#9-aprendizados-e-trabalhos-futuros)

# 1. Descrição e Problema de Negócio

### 1.1 Descrição

A EletroPlaza Store é um conglomerado de lojas presente em diversos países do mundo. Apesar da presença online da marca, as vendas são predominantemente físicas. A multinacional atua com eletrônicos, eletrodomésticos e acessórios em geral, oferecendo produtos acessíveis e de qualidade para uma vasta gama de clientes.

### 1.2 Problema de Negócio

Após diversos investimentos realizados nos setores de vendas, marketing e desenvolvimento de produtos, a empresa teve um grande salto em seu faturamento no último ano. Porém, junto ao faturamento, vieram novos desafios em tentar entender precisamente como as vendas vão continuar se comportando nas próximas semanas.

O ano está chegando ao fim e a Black Friday e o Natal estão se aproximando. As vendas tendem a subir exponencialmente ainda mais nessa época e os dados de comparação do ano passado já não servem mais como termômetro para a empresa. Além disso, a equipe de negócio tem uma meta de faturamento e está preocupada se conseguirá alcançá-la. A urgência em entender os possíveis resultados é grande.

O desafio da sua equipe, formada por cientistas e analistas de dados, é construir uma solução que preveja as vendas semanais das lojas e de seus variados setores para as 5 semanas finais do ano. Deste modo, os gestores da EletroPlaza Store poderão definir planos futuros, organizar estoques, calcular receitas, decidir se farão novos investimentos ou não e onde devem alocar estes novos investimentos, através de campanhas mais eficientes de marketing e vendas.

Os dados disponíveis para a previsão são referentes às vendas realizadas do início do ano vigente até o momento. Cada linha representa um setor específico de uma loja específica naquela determinada semana. São fornecidas diversas informações para análise, inclusive dados externos como temperatura, preço de combustível e taxa de desemprego. Ainda, devido a urgência da solicitação, há diversos dados faltantes na base e a sua equipe não tem tempo hábil o suficiente para coletá-los, devendo buscar soluções alternativas para que mesmo assim consiga realizar previsões eficazes.

**Após determinar os valores das vendas, a missão da sua equipe é enviar um arquivo .csv contendo os identificadores de cada registro e os respectivos valores de venda para a equipe de negócio.**

### 1.3 Expectativas

Os gestores da EletroPlaza Store esperam poder:

- Definir planos;
- Organizar estoques;
- Calcular receitas;
- Decidir se farão novos investimentos ou não;
- Decidir onde devem alocar estes novos investimentos, mediante campanhas mais eficientes de marketing e vendas.

### 1.4 Métrica de Avaliação

A métrica de performance proposta para esse desafio é a raiz quadrada do erro quadrático médio (RMSE), que nada mais é o desvio padrão dos valores residuais, conforme mostra a Equação 1:

![RMSE!](img/rmse.png)

As três equipes que alcançarem o menor RMSE largam em vantagem e passarão para a próxima etapa.

# 2. Base de Dados e Premissas de Negócio

## 2.1 Base de Dados

O conjunto de dados total possui os seguintes arquivos:

- stores.csv - características das lojas
- train.csv - dados de treino
- features.csv - atributos complementares
- test.csv - dados de teste
- sample_submission.csv - exemplo de submissão

Onde cada um apresenta as seguintes features:

**train.csv**

| **Feature**     | **Description**                             |
| --------------- | ------------------------------------------- |
| id              | identificador único do registro             |
| loja            | identificador único da loja                 |
| setor           | setor específico da loja                    |
| data            | semana (mês-dia) em que ocorreram as vendas |
| vendas_semanais | valor total das vendas semanais, em US$     |
| feriado         | se há feriado ou não na semana              |

**features.csv**

| **Feature**            | **Description**                                                                   |
| ---------------------- | --------------------------------------------------------------------------------- |
| loja                   | identificador único da loja                                                       |
| data                   | semana (mês-dia) em que ocorreram as vendas                                       |
| temperatura            | temperatura média da loja na semana, em °C                                        |
| combustivel            | preço médio do combustível na semana, em US$                                      |
| desconto_1             | desconto promocional especial do tipo 1                                           |
| desconto_2             | desconto promocional especial do tipo 2                                           |
| desconto_3             | desconto promocional especial do tipo 3                                           |
| desconto_4             | desconto promocional especial do tipo 4                                           |
| desconto_5             | desconto promocional especial do tipo 5                                           |
| desemprego             | taxa média de desemprego do país na semana                                        |
| feriado                | se há feriado ou não na semana                                                    |
| distancia_competidores | distância média entre as lojas competidoras, em polegadas                         |
| clientes               | quantidades de clientes presentes na loja, em média diária, durante aquela semana |

**stores.csv**

| **Feature** | **Description**                             |
| ----------- | ------------------------------------------- |
| loja        | identificador único da loja;                |
| tipo        | eletrônico, eletrodomésticos ou acessórios; |
| tamanho     | tamanho da loja, unidade não definida.      |

## 2.2 Premissas de Negócio

Nenhuma coluna foi excluida na análise do projeto. Cada linha representa um cliente e cada coluna contém alguns atributos que descrevem esse cliente, além de sua resposta à pesquisa, na qual ele mencionou seu interesse ou não em adquirir o novo produto de seguro.

Para iniciar o projeto é necessário fazer uma junção (merge) dos 3 arquivos apresentados, dessa forma, criando um dataset único onde cada linha representa a venda semanal de um setor de uma loja específica, além de conter especificações da loja e informações complementares que podem ajudar e explicar a variável resposta como o preço do combustível na semana, a taxa de desemprego e principalmente o número de descontos.

# 3. Estratégia de Solução

A estratégia de solução foi a seguinte:

### Passo 01. Descrição dos Dados

Nesse passo foi verificado alguns aspectos do conjunto de dados, como: nome de colunas, dimensões, tipos de dados, checagem de dados faltantes (NA), análise descritiva dos dados e quais suas variáveis categóricas.

### Passo 02. Featuring Engineering

Na featuring engineering foi derivado novos atributos(colunas) baseados nas variáveis originais, possibilitando uma melhor descrição do fenômeno daquela variável.

As principais features criadas nessa etapa foram colunas derivando a data, como dia, mês, semana do mês e semana do ano, além de classificação da loja dependendo do seu tamanho e a soma dos descontos em apenas uma variável afim de diminuir a dimensionalidade do problema.

### Passo 03. Filtragem de Variáveis

Verificando a necessidade de filtrar o conjunto de dados com base em uma variável que não interessa ao projeto em si. Onde nesse projeto em questão nenhuma variável foi deixada de lado.

### Passo 04. Análise Exploratória dos Dados (EDA)

Exploração dos Dados com objetivo de encontrar Insights para o melhor entendimento do Negócio.
Foram feitas também análises univariadas, bivariadas e multivariadas, obtendo algumas propriedades estatísticas que as descrevem, e mais importante respondendo perguntas sobre o negócio.

Através dessa etapas conseguimos chegar em alguns pontos que nos ajudaram a fazer o modelo a performar melhor:

- **Excluir as linhas com vendas semanais nulas ou menores que 1**: Onde essas eram linhas que não conseguiríamos prover, pois se tratam de casos onde os setores das lojas não tiveram vendas, ou tiveram devoluções e estornos. Assim, esse comportamento poderia enviezar o modelo, dessa forma, preferimos excluir essas linhas do dataset.
- **Manter coluna dos descontos**: Haviam poucas semanas com desconto, e eram as últimas semanas do dataset, então no início pensamos em excluir essas colunas. Contudo, durante os eventos de Natal e Black Friday a estratégia dos descontos é muito utilizada para atrair e fidelizar clientes, assim, analisando o dataset de teste, vimos que várias linhas tinham dados de desconto. Decidimos então trabalhar com essa coluna e focar em ajudar ela a explicar o problema, visto que essa pode ser a feature que mais impacta no dataset de teste.

### Passo 05. Preparação dos Dados

Sessão que trata da preparação dos dados para que os algoritmos de Machine Learning possam ser aplicados. Foram realizados alguns tipos de escala e encoding para que as variáveis categóricas se tornassem numéricas.

### Passo 06. Seleção de Variáveis do Algoritmo

Na seleção dos atributos foi realizado um estudo de importância das colunas, no qual os atributos mais significativos foram selecionados para um estudo mais aprofundado com intuito de gerar melhor entendimento e criar features que explicam melhor a variável resposta. Não houve fitragem de variáveis por se tratar de uma competição.

### Passo 07. Modelo de Machine Learning

Realização do treinamento dos modelos de Machine Learning. O modelo que apresentou o menor RMSE diante a base de dados com cross-validation aplicada seguiu adiante para a hiper parametrização das variáveis daquele modelo, visando otimizar a generalização do modelo.

### Passo 08. Hyper Parameter Fine Tuning

Foi encontrado os melhores parâmetros que maximizavam o aprendizado do modelo. Esses parâmetros foram definidos com base no método de RandomSearch.

### Passo 9. Deploy do Modelo em Produção

Com o modelo e seus parâmetros escolhidos, o mesmo treinou a base completa dos dados (sem divisão de treino e teste) e foi usado para prever os valores para a submissão. Dessa forma, cada vez que um modelo era treinado e apresentava uma performance melhor (menor erro RMSE) o mesmo era usado para prever em cima dos dados de submissão e com isso o arquivo era enviado para o Kaggle.

### Passo 10. Apresentação Final

Como a equipe classificou-se para a final (ficamos no TOP 3), foi criado um slide com a explicação dos métodos utilizados para chegar na performance.

# 4. Exploration Data Analysis

## 4.1 Análise Univariada

- Variáveis Numéricas: o histograma abaixo mostra como está organizada a distribuição das variáveis numéricas do nosso conjunto de dados. Mostra a contagem de cada variável numérica do dataset.

![Numerical-Variables!](img/uni.png)

## 4.2 Análise Bivariada

### H2 - Pessoas mais velhas tem uma tendência maior de se interessarem pelo seguro.

**FALSO** Quanto maior a idade, MENOR a tendência de se interessar pelo seguro de automóveis.

- Nos gráficos da 1ª linha podemos ver que a base apresenta uma maior quantidade de clientes entre 20 e 30 anos e mostra uma linha de tendência crescente, quanto maior a idade, menos amostrar na base.
- Nos gráficos da 2ª linha podemos ver a relação da idade de clientes que aceitariam o seguro, com a predominância de clientes entre 40 e 50 anos, com uma linha de tendência decrescente, quanto maior a idade menos clientes.
- Dessa forma, entre os clientes que aceitariam o seguro há uma predominância de clientes mais velhos, contudo, a linha de tendência mostra que quanto mais idade, há menos clientes que aceitariam o seguro.

![H2!](img/H2.png)

### H5 - Quanto mais velho o veículo, maior a chance da pessoa se interessar pelo seguro.

**VERDADEIRO** - A proproção de pessoas que se interessam pelo seguro é maior quando o veículo tem mais de 2 anos, portanto, quando mais velho o veículo, maior a chance da pessoa se interessar pelo seguro.

- No 1º gráfico observa-se que a quantidade de amostras de clientes com veículo com menos de um ano, e entre um ano e dois, são bem parecidos. A quantidade de clientes com veículos com mais d 2 anos é de 4,2%.
- Entre aqueles que se interessam pelo seguro, há uma predominância de clientes com veículos entre um e dois anos. Contudo, a proporção de veículos com mais de dois anos apresentou um crescimento considerável.
- No 3º gráfico vemos que a proporção de clientes interessados pelo seguro é maior naqueles que com veículos com mais de dois anos.

![H5!](img/H5.png)

### H7 - Quanto mais a pessoa paga de Annual Premium, maior a chance dela se interessar em ter um seguro.

**FALSO** - Quanto mais a pessoa paga no Annual Premium, menos ela se interessa em adquirir o seguro.

- Na esquerda vemos os gráficos da relação da quantidade de clientes com o quanto pagam de Annual Premium.
- Na direita vemos os gráficos da relação da quantidade de clientes que se interessam pelo com o quanto pagam de Annual Premium.
- Os primeiros gráficos mostram o total de Annual Premium, os segundos dão um zoom nos que pagam menos que 100.000 e os terceiros gráficos mostram os que pagam mais que 100.000.
- A maioria do clientes pagam menos que 55.000 de Annual Premium, e quanto mais os clientes pagam, menor a quantidade deles na base e menor o interesse no seguro de automóveis.

![H7!](img/H7.png)

### H8 - Quanto maior o tempo que a pessoa tem o seguro de vida, maior a chance dela se interessar pelo seguro de carro.

**FALSO** - Depois de 15 dias, há uma tendência negativa, portanto, quanto maior o vintage, menor a chance da pessoa não se interessar pelo seguro de automóveis.

- Nos gráficos da 1ª linha vemos a quantidade de clientes que se interessam pelo seguro segmentado por uma faixa de dias que tem o seguro se saúde.
- Nos gráficos da 2ª linha mostra os mesmos valores, tirand os clientes com vintage entre 0 e 15 dias, que apresentam uma baixa quantidade na base e isso enviésa a análise.
- Assim, percebemos que a quantidade de clientes em cada faixa de vintage é bem parecido, contudo, quando olhamos para a tendência do gráfico, vamos uma tendência de queda, onde, quando maior a faixa de vintagem, menos clientes se interessam pelo seguro.

![H8!](img/H9.png)

### Tabela de Insights

| Hipóteses                                                                                                   | Condição   |
| :---------------------------------------------------------------------------------------------------------- | :--------- |
| H1 - Das pessoas interessadas no seguro, a maioria são homens.                                              | Verdadeira |
| H2 - Pessoas mais velhas tem uma tendência maior de se interessarem pelo seguro.                            | Falsa      |
| H3 - Das pessoas interessadas no seguro, a maioria tem licença para dirigir.                                | Verdadeira |
| H4 - Das pessoas interessadas no seguro, a maioria tem licença para dirigir.                                | Verdadeira |
| H5 - Quanto mais velho o veículo, maior a chance da pessoa se interessar pelo seguro.                       | Verdadeira |
| H6 - Pessoas que sofreram batida de carro no passado, estão mais interessadas em adquirir o seguro.         | Verdadeira |
| H7 - Quanto mais a pessoa paga de Anual Premium, maior a chance dela se interessar em ter um seguro.        | Falsa      |
| H8 - Quanto maior o tempo que a pessoa tem o seguro de vida, maior a chance dela se interessar pelo seguro. | Falsa      |

# 5. Seleção do Modelo de Machine Learning

Os seguintes algoritmos de Machine Learning foram aplicados:

- KNN Classifier;
- Logistic Regression Classifier;
- Extra Trees Classifier;
- Random Forest Classifier;
- XGBoost Forest Classifier;
- Gaussian Naive Bayes;

O método de cross-validation foi utilizado em todos os modelos.

# 6. Performance do Modelo

Para medir o desempenho dos modelos, usaremos o método de validação cruzada que evita que o modelo seja superajustado quando o modelo recebe alguns dados que nunca viu antes (garantindo a generalização). O @K para as métricas da abordagem Ranking-To-Learn é 20.000, sendo a quantidade mínima de tentativas que a empresa quer fazer para atingir o máximo de acertos na base.

A real performance dos modelos utilizando método CROSS-VALIDATION.

| Model Name             | Accuracy Balanced     | Precision @K Mean     | Recall @K Mean       | ROC AUC Score        | Top K Score           |
| ---------------------- | --------------------- | --------------------- | -------------------- | -------------------- | --------------------- |
| **XGBClassifier**      | **0.5367 +/- 0.0013** | **0.2988 +/- 0.0019** | **0.8005 +/- 0.005** | **0.839 +/- 0.0011** | **0.8686 +/- 0.0003** |
| RandomForestClassifier | 0.5452 +/- 0.0013     | 0.2914 +/- 0.0015     | 0.7806 +/- 0.0039    | 0.8306 +/- 0.0009    | 0.865 +/- 0.0004      |
| GaussianNB             | 0.784 +/- 0.0003      | 0.29 +/- 0.0021       | 0.7769 +/- 0.0056    | 0.8259 +/- 0.0021    | 0.6384 +/- 0.0002     |
| ExtraTreesClassifier   | 0.5515 +/- 0.0018     | 0.2875 +/- 0.0017     | 0.7703 +/- 0.0046    | 0.825 +/- 0.0009     | 0.8599 +/- 0.0003     |
| LogisticRegression     | 0.5 +/- 0.0           | 0.2754 +/- 0.0019     | 0.7379 +/- 0.005     | 0.817 +/- 0.0023     | 0.8776 +/- 0.0        |
| KNeighborsClassifier   | 0.5489 +/- 0.0006     | 0.2743 +/- 0.0014     | 0.7349 +/- 0.0037    | 0.7811 +/- 0.0012    | 0.8609 +/- 0.0004     |

Além do Cross-Validation, foi comparado o ganho de todos os modelos testados e o resultado pode ser visto na imagem abaixo:

![Comparação!](img/comparacao.png)

O Modelo Final escolhido foi o XGBoost Classifier, devido ao seu melhor Recall com 20.000 tentativas e sua melhor performance para atingir um total de 80% de acertos. Os parâmetros da performance final estão abaixo.

| Model Name        | Accuracy Balanced | Precision @K Mean | Recall @K Mean | ROC AUC Score | Top K Score |
| ----------------- | ----------------- | ----------------- | -------------- | ------------- | ----------- |
| **XGBClassifier** | **0.5015**        | **0.3361**        | **0.7253**     | **0.8607**    | **0.8779**  |

# 7. Pontuação Final

Com base no método atual de previsão de ordenação de clientes é possível analisarmos a diferença de performance entre o modelo utilizado (Modelo Aleatório) e o modelo proposto XGBoost Classifier.

Considerando o modelo aletório como a ordenação inicial da lista. A lista apresenta no total **9.526** clientes interessados em adquirir o seguro de automóveis.

## 7.1 Qual é a porcentagem de interesse dos clientes em adquirir um seguro de automóvel que a equipe de vendas conseguirá atingir com 20.000 ligações?

Para responder essa pergunta foi plotado o a curva de ganho acumulado e a curva lift, destacando o ponto que representa 20.000 clientes do total da base.

![Q1!](img/q1.png)

Através disso percebemos:

**- Entrando em contato com os Top 20.000 clientes da lista (26,24% de toda a base), eu encontraria 72,53% de todos aqueles interessados pelo seguro.**

**- Selecionando apenas os Top 20.000 clientes da lista, o modelo proposto é aproximadamente 2,75 melhor que o modelo aleatório.**

Se considerarmos que o preço fixo do seguro é de U$ 2.000,00 por ano e não colocando na questão o custo de cada ligação para a empresa teremos os seguintes resultados:

| Modelo                     | Pessoas Interessadas | % de Acertos | Receita         |
| -------------------------- | -------------------- | ------------ | --------------- | -------------- |
| Modelo Aleatório           | 2.448                | 26,45%       | U$4.895.653,00  |
| XGBoost Classifier         | 6.713                | 72,53%       | U$13.426.593,00 |
| Diferença entre os modelos | 4.265                | -----        | -----           | U$8.530.940,00 |

**- Dessa forma, caso a empresa queira selecionar apenas os Top 20.000 da lista, o modelo proposta atinge 4.265 mais clientes interessados que o modelo aletório, gerando U$8,5 Milhões a mais de receita anual.**

## 7.2 Qual é a porcentagem de interesse dos clientes em adquirir um seguro de automóvel que a equipe de vendas conseguirá atingir com 40.000 ligações?

Para responder essa pergunta foi plotado o a curva de ganho acumulado e a curva lift, destacando o ponto que representa 40.000 clientes do total da base.

![Q2!](img/q2.png)

Através disso percebemos:

**- Entrando em contato com os Top 40.000 clientes da lista (52,48% de toda a base), eu encontraria 99,34% de todos aqueles interessados pelo seguro.**

**- Selecionando apenas os Top 40.000 clientes da lista, o modelo proposto é aproximadamente 1,9 melhor que o modelo aleatório.**

Se considerarmos que o preço fixo do seguro é de U$ 2.000,00 por ano e não colocando na questão o custo de cada ligação para a empresa teremos os seguintes resultados:

| Modelo                     | Pessoas Interessadas | % de Acertos | Receita         |
| -------------------------- | -------------------- | ------------ | --------------- | -------------- |
| Modelo Aleatório           | 4.825                | 52,13%       | U$9.649.490,00  |
| XGBoost Classifier         | 9.195                | 99,34%       | U$18.390.158,00 |
| Diferença entre os modelos | 4.370                | -----        | -----           | U$8.740.668,00 |

**- Dessa forma, caso a empresa queira selecionar apenas os Top 40.000 da lista, o modelo proposta atinge 4.370 mais clientes interessados que o modelo aletório, gerando U$8,7 Milhões a mais de receita anual.**

## 7.3 Quantas ligações a equipe de vendas precisa para atingir 80% dos clientes interessados em contratar um seguro de automóvel?

Para responder essa pergunta o contrário foi feito. A priori podemos observar a curva de ganho com a marcação no ganho de 80%, conrrespondendo a 30,25% do total da base.

![Q3!](img/q3.png)

Com isso, podemos perceber:

| Modelo                     | Qtde de ligações | % da Base |
| -------------------------- | ---------------- | --------- |
| Modelo Aleatório           | 61.161           | 80,24%    |
| XGBoost Classifier         | 23.057           | 30,25%    |
| Diferença entre os modelos | 38.104           | -----     |

**- Dessa forma, caso a empresa tenha como objetivo alcançar 80% do total de interessados, no modelo proposto ela alcançará esse resultado com 38.104 ligações a menos que o modelo aletório, gerando muita economia e gastando menos tempo da sua equipe de vendas.**

# 8. Conclusão

Neste projeto, todas as etapas necessárias foram realizadas para implementar um projeto completo de Ciência de Dados em um ambiente de produção. Foi utilizado o método de gerenciamento de projetos denominado CRISP-DM/DS e obteve-se um desempenho satisfatório utilizando o modelo Classificador XGBoost para ranquear os clientes com maior propensão de obtenção do novo seguro: o seguro de automóveis.

Alguns Business Insights foram gerados durante a Análise Exploratória de Dados que ajudaram o CEO, a equipe de negócios e o cientista de dados a entender melhor o negócio. Diante desses resultados, o projeto atingiu seu objetivo de encontrar uma solução assertiva de ranqueamento de clientes de forma a otimizar a eficácia da equipe de vendas no contato com seus clientes.

Com o modelo em produção, a expectativa é que ele seja pelo menos 2,5x mais eficaz em fazer com que os clientes adquiram o novo produto de seguro de automóveis. Para atingir cerca de 80% dos interessados em adquirir o seguro auto é preciso atingir apenas 30% dos clientes da base de dados.

# 9. Aprendizados e Trabalhos Futuros

**Aprendizados**

- Esse problema de classificação foi resolvido de uma forma diferente dos normais problemas desse tipo, ao invés de classificar os resultados, utilizamos uma abordagem de Learn To Rank, afim de ordenar os clientes baseado em sua propensão para se interessar ao novo produto.

- Como é uma abordagem diferente, outras métricas foram aplicadas aos modelos, e apesar de seus resultados aproximados, o escolhido foi o que apreentou a melhor performance para resolver o problema de negócios. Assim, o aprendizado nesse quesito foi, mesmo que as métricas sejam boas, o melhor modelo será aquele que resolve melhor o problema, que nesse caso era atingir o máximo de clientes interessados dos Top 20.000 clintes da base.

- A Análise Exploratória de Dados se demonstrou uma das etapas mais importantes do projeto, pois é nessa parte que podemos encontrar Insights de Negócio que promovem novos conhecimentos e até contradições que nos fazem repensar o negócio como um tudo. Essa análise também fornece ao cientista de dados uma "direção" de como melhorar seu modelo, por meio da criação de novas features e diferentes tipos de abordagem.

**Trabalhos Futuros**

- Criar novas features, afim de explicar com mais eficiência o os fenômenos do problema e consequentemente gerar resultados melhores.
- Testar diferentes Encoders na preperação dos dados.
- Aplicar o balanceamento dos dados e entender como isso influencia na resolução desse problema.
- Gerar mais insights relacionando diferentes colunas, isso ajudará a criar novas features.
