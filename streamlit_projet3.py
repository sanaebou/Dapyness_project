#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 22:49:08 2021

@author: sanaeboulahtit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#tests statistiques
import statsmodels.api 
from scipy.stats import pearsonr



#modèles arima et sarima
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


df = pd.read_csv('data.csv',encoding = "ISO-8859-1")
df_sale = pd.read_csv('df_sale.csv',encoding = "ISO-8859-1")


st.title("Projet DaPyness - Estimation de ventes e-commerce")

page = st.sidebar.radio(label = "", options = ['Présentation',
                                               'Etude du jeu de données',
                                               'Modèle de prévision des ventes',
                                               'Conclusion'])

#première page
if page == "Présentation du projet":
    
     
    st.markdown("""
                 Cette application Streamlit a été réalisée dans le cadre 
                 du projet de l'équipe Dapyness issue de la **promotion 
                 Data Analyst de Datascientest de janvier 2021**.
                 
                 ***L'équipe Dapyness est composée de David Sitbon, 
                 Sanae Boulahtit, Axel Durimele, et Naia Nandaboure.***
                 
                 L’objectif de ce projet est d’estimer l’évolution 
                 du volume de ventes d'un site e-commerce 
                 en utilisant les données publiques disponibles.
                 """)
    
    st.markdown("""
                 *Voici un aperçu du jeu de données original :*
                 """)
    st.write(df.head(50))

    st.markdown("""
                Le jeu de données original est constitué de 
                **541 909 lignes et 8 colonnes**, de type **'object'** 
                (renvoie un objet vide), **'float'** (renvoie un nombre à virgule) 
                ou **'int'** (renvoie un entier).
                
                Nous avons repéré des valeurs manquantes dans les colonnes 
                'Description' et 'CustomerID'. Nous les traiterons dans la suite.
                """)      
                
    st.markdown("""
                *Voici un aperçu de df.describe() :* 
                """)
    st.write(df.describe())           
                
    st.markdown("""         
                La colonne 'Quantity' contient des valeurs très élevées 
                et très basses, 50 % des valeurs sont comprises entre 1 et 10.
                
                La colonne 'UnitPrice' contient également des valeurs 
                très élevées et très basses, 50 % des valeurs sont comprises 
                entre 1.25 et 4.13.
                """)
    
    st.markdown("""
                ***Ainsi, nous avons une vue d'ensemble du jeu de données 
                à notre disposition. Avant de pouvoir construire un modèle 
                de prédictions des ventes, nous devons étudier le jeu de données, 
                le nettoyer et réorganiser les données, si besoin.***
                """)
                
    st.header('**Traitement des doublons**')
                    
    st.markdown("""
                Nous avons repéré l'existence de plusieurs **lignes identiques**.
                Cependant, nous éméttons l'hypothèse selon laquelle 
                **il ne s'agit pas de doublons**. 
                En effet, un article peut être entré plusieurs 
                fois dans un panier, et cela pourrait expliquer 
                la présence de plusieurs lignes identiques dans le dataset.
                
                Exemple concret : une valeur de la variable CustomerID
                peut se répéter dans le dataset et cela est considéré comme 
                des doublons alors qu'en réalité cela correspond simplement 
                à la situation où un acheteur commande plusieurs articles.
                """)
                
    st.header('**Gestion des valeurs manquantes**')
                
    st.markdown("""
                Nous avons repéré la présence de NaN dans les colonnes 
                'Description' et 'CustomerID'.
                
                Nous avons décidé, d'une part, de **remplacer les NaN de CustomerID
                par une nouvelle catégorie "Client non identifié".** 
                Il y avait près de 25% de NaN dans la variable CustomerID,
                donc leur suppression n'est pas envisageable.
                
                Et, d'autre part, nous avons **supprimer les lignes contenant une NaN
                dans la variable 'Description'.** En effet, cele ne représentait 
                qu'1 % des valeurs de cette variable alors leur suppression ne biaisera
                par notre étude.
                """)
                
    st.header('**Création de nouvelles colonnes**')
                
    st.markdown("""
                 Dans le Dataset, nous avons identifié qu'il y a plusieurs types 
                d'opérations (ventes, annulation, frais, etc..).
                C'est pourquoi, **nous avons créé une nouvelle colonne 
                'InvoiceType' ans laquelle nous catégorisons les lignes 
                selon leur type** : vente, annulations, dettes, frais, échantillons
                
                Par ailleurs, nous avons également décidé **d'ajouter plusieurs
                colonnes qui correspondent à la periode de la commande** 
                afin de rendre **exploitable** et **analysable** les données 
                et les informations de la variable **InvoiceDate**, 
                en plusieurs étapes, à l'aide de la méthode .to_datetime.
                Nous sommes donc passé d'une colonne InvoiceDate de format 
                "12/1/2010 8:26" à 4 colonnes pour séparer l'information 
                de l'année (Year), du mois (Month), du jour (Day) et 
                l'horaire (Time).       
                """)
                
    st.markdown("""
               Voici un aperçu du nouveau Data set df_sale
               """)
    st.write(df_sale.head(50))
                     
    st.markdown("""             
                Dans la section suivante, nous avons étudié en profondeur
                les données du dataset pour mieux appréhender 
                la problématique de modélisation d'un programme de prédiction
                des ventes.
                """)



if page == 'Etude du jeu de données':
    
    st.markdown("""
                Durant notre étude, nous avons étudié plusieurs relations et nous avons représenter cela à l'aide de graphiques. 
                Nous avons introduit ici les datavisualisations les plus intéressantes.
      
                """)
    
        
    
    st.markdown("""
                Nous avons trouvé intéressant d'étudier le lien que pouvait avoir la variable *'Quantity'*
                avec les différents attributs de la date, à savoir le mois, le jour du mois, le jour de la semaine, 
                la semaine de l'année ainsi que l'heure. On y trouve quelques dépendances statistiques.
                
                Voici l'étude approfondie de ces relations :
                """)
          
    # datavisualisations avec les attributs dates            
    relations = ['Relation Quantité - Mois', 'Relation Quantité - Semaine', 'Relation Quantité - Jour du mois',
              'Relation Quantité - Jour de la semaine', 'Relation Quantité - Heure']
    choix_relations = st.radio("", options = relations)            
    
    #1ere visualisation --> jour du mois
    if choix_relations == 'Relation Quantité - Jour du mois' :
        vis =df_sale[['Quantity','Day']].groupby('Day').sum().reset_index()
        sns.barplot(x='Day',y='Quantity', data=vis).set(title='Volume de vente en fonction du jour du mois');
        axes = plt.gca()
        axes.set_xlabel('')
        axes.set_ylabel('Volume de vente')
        fig =plt.gcf()
        fig.set_size_inches(10,5)
        
        st.pyplot(fig)
        
        st.markdown("""
                    On remarque une saisonalité et une tendance à la baisse.
                    Nous remarquons qu'il y a moins de ventes en début du mois et beaucoup moins 
                    de ventes à la fin du mois. 
                    
                    Le volume de vente semble avoir un lien avec le jour du mois. Etudions 
                    la relation statistique entre ces deux variables : 
                    
                    
                    """)
        
        # Etude de la relation statistique entre Quantity et Day (test de l'anova)
        result = statsmodels.formula.api.ols('Quantity ~ Day', data = df_sale).fit()
        table = statsmodels.api.stats.anova_lm(result)
        
        st.write(table)
        
        st.markdown("""
                    La p-valeur est supérieur à 5%. Ainsi, on ne rejette pas l'hypothèse d'indépendance
                    et on conclut que la Quantité ne dépend pas du jour d'achat du mois.
                    """)


    #2eme visualisation --> jour de la semaine
    if choix_relations == 'Relation Quantité - Jour de la semaine' :
        vis =df_sale[['Quantity','DayOfWeek']].groupby('DayOfWeek').sum().reset_index()
        sns.barplot(x='DayOfWeek',y='Quantity',  palette='husl',data=vis).set(title='Volume de vente par jour de la semaine');
        axes = plt.gca()
        axes.set_xlabel('')
        axes.set_ylabel('Volume de ventes')
        axes.xaxis.set_ticklabels(['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Dimanche']);
        fig =plt.gcf()
        fig.set_size_inches(10,5)
        
        st.pyplot(fig)
        
        st.markdown("""
                    On observe qu'il n'y a pas de ventes le samedi,
                    Les jeudi et mardi sont les jours comptant le plus de vente.
                    Le dimanche est le jour comptabilisant le moins de ventes.
                    le volume de vente semble avoir un lien avec le jour de la semaine.
                    """)
                    
        # Etude de la relation statistique entre Quantity et DayOfWeek (test de l'anova)
        result = statsmodels.formula.api.ols('Quantity ~ DayOfWeek', data = df_sale).fit()
        table = statsmodels.api.stats.anova_lm(result)
    
        st.write(table)
        
        st.markdown("""
                    La p-valeur est inférieure à 5%. Ainsi, on rejette l'hypothèse d'indépendance
                    et on conclut que la Quantité a un lien de dépendance avec le jour de la semaine.
                    """)

    #3eme visualisation --> mois
    if choix_relations == 'Relation Quantité - Mois' :
        vis =df_sale[['Quantity','Month']].groupby('Month').sum().reset_index()
        sns.barplot(x='Month',y='Quantity',  palette='husl',data=vis).set(title='Volume de ventes par mois');
        axes = plt.gca()
        axes.set_xlabel('')
        axes.set_ylabel('Volume de ventes')
        axes.xaxis.set_ticklabels(['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aou','Sep','Oct','Nov','Dec']);
        fig =plt.gcf()
        fig.set_size_inches(10,5)
        
        st.pyplot(fig)
        
        st.markdown("""
                    Nous observons une hausse des ventes à partir du mois de septembre. 
                    Les ventes atteignent un pic en novembre. Ce qui peut s'expliquer par l'approche de la période de fêtes.
                    
                    Le volume de vente semble dépendre du mois de l'année.
                    """)
        
        result = statsmodels.formula.api.ols('Quantity ~ Month', data = df_sale).fit()
        table = statsmodels.api.stats.anova_lm(result)
        
        st.write(table)
        
        st.markdown("""
                    La p-valeur est supérieur à 5%. Ainsi, on ne rejette pas l'hypothèse d'indépendance
                    et on conclut que la Quantité ne dépend pas du mois d'achat de l'année.
                    """)
        
    #4ème visualisation --> semaines
    if choix_relations == 'Relation Quantité - Semaine' :
        volume_par_semaine = df_sale[['Quantity','Week']].groupby('Week').sum().reset_index()
        sns.barplot(x='Week',y='Quantity', data=volume_par_semaine).set(title='Volume de vente par semaine de l année');
        axes = plt.gca()
        axes.set_xlabel('')
        axes.set_ylabel('Volume de vente')
        fig =plt.gcf()
        fig.set_size_inches(10,5)
        
        st.pyplot(fig)
        
        st.markdown("""
                    Nous observons une tendance à la hausse à la fin de l'année.' 
                    Les ventes atteignent un pic en semaine 49 de l'année puis décroît assez 
                    rapidement les deux dernières semaines de l'année.
                    
                    Le volume de vente semble dépendre de la semaine de l'année.
                    """)
        
        result = statsmodels.formula.api.ols('Quantity ~ Week', data = df_sale).fit()
        table = statsmodels.api.stats.anova_lm(result)
        
        st.write(table)
        
        st.markdown("""
                    La p-valeur est supérieur à 5%, ce qui ne nous permet pas de rejetter l'hypothèse d'indépendance.
                    On en déduit que la Quantité ne dépend pas de la semaine de l'année.
                    """)
    
    #5ème visualisation --> heures
    if choix_relations == 'Relation Quantité - Heure' :
        volume_par_heure = df_sale[['Quantity','Hour']].groupby('Hour').sum().reset_index()
    
        sns.barplot(x='Hour',y='Quantity', data=volume_par_heure).set(title='Volume de ventes par heure de la journée');
        axes = plt.gca()
        axes.set_xlabel('')
        axes.set_ylabel('Volume de ventes')
        fig =plt.gcf()
        fig.set_size_inches(10,5)
        
        st.pyplot(fig)
        
        st.markdown("""
                    Nous remarquons que les ventes ont lieu uniquement de 6h à 20h.
                    Aucunes ventes n'est réalisée à partir de 20h et ce, jusqu'à 6h. 
                    
                    Il semblerait que cela corresponde à des horaires de bureau ou des horaires d'ouverture d'une boutique physique.
                    Hors, nos données correspondent normalement à des ventes effectuées en ligne, et ceux, dans plusiseurs pays différents.
                    Les ventes devraient logiquement avoir lieu à n'importe quelle heure de la journée.
                    
                    IL semble y avoir un lien entre la quantité er l'heure de vente.
                    """)
        
        result = statsmodels.formula.api.ols('Quantity ~ Hour', data = df_sale).fit()
        table = statsmodels.api.stats.anova_lm(result)
        
        st.write(table)
        
        st.markdown("""
                   La p-valeur est inférieure à 5%. Ainsi, on rejette l'hypothèse d'indépendance.
                   On conclut que la Quantité a un lien de dépendance avec l'heure de vente de la journnée.
                   """)
    
    
    
    st.header('Autres visualisations intéressantes')
    
    st.markdown("""
                Tout d'abord, nous avons représenté le volume de vente en fonction de la variable *'InvoiceDate'* qui représente la date :
                """)
    
    #datavisualisation : serie temporelle
    fig1, ax = plt.subplots()
    vis1 =df_sale[['InvoiceNo','Date']].groupby('Date').count().reset_index()
    sns.lineplot(vis1['Date'], vis1['InvoiceNo'], linewidth = 1, color = 'darkblue')
    plt.ylabel('Nombre de ventes', fontsize=20)
    plt.title('Nombre de ventes par jour', fontsize=25)
    plt.xticks(vis1['Date'][::30],fontsize=12, rotation = 45)
    fig1.set_size_inches(18,12)
    st.pyplot(fig1)
    
    st.markdown("""
                Il semblerait que l'on soit confronté à une série temporelle avec des saisonalités.
                
                Cela est très intéressant et nous permettra d'explorer des méthodes de modélisations liées aux séries temporelles.
                """)
    st.markdown("""
                Nous avons également étudié le volume de ventes en fonction du pays d'où provient l'achat :
                """)
    
    
    #datavis : volume de ventes par payss
    #Volume de ventes par pays
    orders_by_country = df_sale.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
    #Plot
    fig2, ax = plt.subplots()
    orders_by_country.plot(kind = 'bar')
    plt.xlabel('Pays')
    plt.ylabel('Volume de ventes')
    plt.title('Volume de ventes par pays', fontsize=16)
    st.pyplot(fig2)
    
    st.markdown("""
                Nous remarquons que la grande majorité des ventes se concentre au Royaume-Uni.
                
                Nous pouvons exclure cette données pour avoir plus de visibilité sur les données des autres pays :
                """)
    
    fig3, ax = plt.subplots()
    quantity_by_country = df_sale.groupby('Country')['Quantity'].count().sort_values(ascending=False)
    del quantity_by_country['United Kingdom']
    #Plot
    quantity_by_country.plot(kind = 'bar')
    plt.xlabel('Pays')
    plt.ylabel('Volume de ventes')
    plt.title('Volume de ventes par pays', fontsize=16)
    st.pyplot(fig3)
    
    st.markdown("""
                Nous voyons que le Royaume-Unis est suivi par l'Allemagne et la France. 
                
                La quantité semble dépendre du pays de provenance de l'achat, ce qui se vérifie avec le test statistique d'Anova suivant :
                """)
    
    import statsmodels.api 
    result = statsmodels.formula.api.ols('Quantity ~ Country', data = df_sale).fit()
    table1 = statsmodels.api.stats.anova_lm(result)
    st.write(table1)
    
    st.markdown("""
                Au vu des résulats du test, nous rejetons l'hypothèse d'indépendance des deux variables. 
                
                Le volume de vente dépend bien du pays d'origine de l'achat.
                """)

    st.markdown("""
                Nous avons ainsi étudié les graphiques les plus intéressants obtenus. 
                Nous avons pu voir que notre jeu de données cache une série temporelle. 
                Ce qui nous conduit à explorer des modèles de prévisions de ventes en nous basant sur les algorithmes ARIMA et SARIMA. 
                
                Nous voyons cela dans la section suivante.
                """)
    
if page == 'Modèle de prévision des ventes':    
    
    st.markdown("""
                Dans cette section, nous introduisons un modèle de prédictions de ventes. 
                
                Notre variable cible est quantitative. 
                Nous avons ainsi exclu tous les modèles s’apparentant à de la classification dans notre étude.
                En premier lieu, nous nous sommes tout naturellement tournés vers des modèles de régression.
                
                Notre jeu de données ne contient initialement qu’une seule variable explicative de nature quantitative : 
                    le prix unitaire du produit vendu. 
                Dans un premier temps, nous avons donc tenté de créer un modèle de régression simple 
                entre la variable représentant la quantité et la variable correspondant 
                au prix unitaire. Nous avons obtenu un score non exploitable avec ce modèle. 
                Ce qui peut s’expliquer par le fait qu’une des hypothèses n’était pas vérifiée : 
                    la linéarité entre la variable cible et la variable explicative. 
                En effet l’étude visuelle de la relation entre ces deux variables nous a permis de conclure 
                qu’il n’y avait aucune relation de linéarité.
                
                """)  
    df_sale2=df_sale[df_sale["Quantity"]< 7000]
    df_sale2=df_sale2[df_sale2["UnitPrice"]< 300]
    
    fig, ax = plt.subplots()
    plt.scatter(df_sale2.Quantity, df_sale2.UnitPrice, c = "blue",  s=10)
    fig.set_size_inches(6,3)
    st.pyplot(fig)


    st.markdown("""
                Nous avons ensuite créé de nouvelles features pour ainsi les introduire dans un modèle de régression.
                Tous ces modèles n'ont pas été performants et ne nous ont pas permis 
                d’obtenir des résultats satisfaisants. 
                 
                Nous nous sommes ainsi tournés vers une autre méthode plus adaptée à notre problèmatique : **l'étude de série temporelle**
                """)

    st.header('**Preprocessing**')

    st.markdown("""
                Après avoir étudié et compris notre jeu de données, 
                nous avons transformé notre table pour ainsi l'utiliser dans des modèles **ARIMA** et **SARIMA**.
                
                Nous avons agrégé les données de volume de vente par jour et nous avons transformé 
                notre jeu de données en série temporelle.
                Nous avons ensuite stationnarisé la série pour pouvoir ainsi créer notre
                modèle de prédiction des ventes.
                """)

    #On repart de nos données quantity agrégées par jour
    df_daily = df_sale.groupby(by=['Date'])['Quantity'].sum().reset_index()
    df_daily.head()
    #conversion de la date en format date
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    #on place la date en index
    df_daily.index=df_daily.Date
    df_daily=df_daily.drop(['Date'], axis=1)
    #on enregistre df_daily sous forme de série
    df_daily = pd.Series(data=df_daily['Quantity'])

    st.write(df_daily.head())


    #rpz de la série temporelle
    st.markdown("""
                *Représentation de la série temporelle* :
                """)
    fig, ax = plt.subplots()
    plt.plot(df_daily)
    fig.set_size_inches(6,3)
    st.pyplot(fig)

    st.markdown("""
                Avant de créer un modèle SARIMA, il faut nous assurer que la série est stationaire. 
                Dans le cas où cette condition n'est pas vérifiée, nous devons la rendre stationnaire.
                Nous utilisons le test augmenté de Dickey-Fuller avec HO : la série a une racine unitaire, 
                ce qui indiquerait qu'elle est non stationnaire et H1 l'hypothèse contraire.
               
                Ici, nous obtenons une p-valeur de 0.756 largement supérieure à 5%.
                --> On ne rejette pas l'hypothèse H0. La série est non stationnaire. IL faut donc la stationariser.
                
                
                Nous effectuons une décomposition de la série temporelle et nous observons une tendance
                à la hausse : nous avons un modèle multiplicatif et non additif.
               """)
               
    #décomposition saisonal multiplicative
    d_seasonal = plt.imread("d_seasonal.jpg")
    st.image(d_seasonal)

    st.markdown("""
                Nous commençons par utilisier la transformation en log afin de nous ramener à un modèle additif.
                
                La première étape d'une prévision à l'aide des modèles SARIMA est de stationnariser notre série temporelle, 
                autrement dit, nous devons estimer les paramètres D et d de notre modèle *SARIMA(p,d,q)(P,D,Q)s*.

                On se sert des autocorrélations simples. Quel que soit le type de processus, si
                l'autocorrélation simple décroît assez rapidement vers 0 --> le processus est stationnaire.
                
                Après avoir appliqué une différenciation d'ordre 1 ainsi qu'une différenciation
                de période 6 sur notre série temporelle, nous obtenons une série temporelle stationnaire.
                
                Nous avons ainsi d=D=1 et k=6 car on a différencié une fois à l'ordre 1 et une fois à l'ordre 6 (car saisonalité de 6)
                
                Il nous reste à chercher les paramètres P,Q,p et q
                pour ainsi avoir un processus SARIMA qui pourrait modéliser au mieux notre série temporelle. 
                Pour cela, nous utiliserons les autocorrélogrammes simples et partiels '''
                """)

    df_dailylog = np.log(df_daily)
    df_dailylog_1 = df_dailylog.diff().dropna()
    df_dailylog_2 = df_dailylog_1.diff(periods = 6).dropna()
    
    
    #graphs dautocorrélation
    autocorrelation = plt.imread("autocorrelation.jpg")
    st.image(autocorrelation)

    st.markdown("""
                La zone bleutée représente la zone de non significativité des autocorrélogrammes. 
                Les autocorrélogrammes simple et partiel s'annulent à partir du rang 2.
                
                On a p=q=2-1=1 
                
                Concernant la saisonalité, on distingue un petit pic pour chaque autocorrélogramme au rang 6,
                puis rien de significatif aux rangs supérieurs. Les paramètres p,q,P et Q prendront donc tous initialement la valeur 1.

                """)

    st.header('**Modélisation**')

    st.markdown("""
                Après avoir séparé notre jeu de données en un échantillon train et un échantillon test,
                nous procédons à la création du modèle.
                
                Nous créons ainsi un modèle SARIMA(1,1,1)(1,1,1,6). 
                Le paramètre p étant non significatif, nous le retirons du modèle, ce qui nous donne :
                    
            
                """)

    # Daily: Train & Test Split
    series_date=df_daily
    split_time = 250
    time_d=np.arange(len(df_daily))
    train_d=series_date[:split_time]
    test_d=series_date[split_time:]
    timeTrain_d = time_d[:split_time]
    timeTest_d = time_d[split_time:]

    s_model = SARIMAX(endog=train_d , order=(0, 1, 1), seasonal_order=(1, 1, 1, 6), trend='t')
    s_model_fit=s_model.fit()
    st.write(s_model_fit.summary())

    st.markdown("""
                
                Il nous restait ensuite à vérifier certaines hyptohèses : celle que le résidu est un bruit blanc et qu’il est distribué normalement.
                
                Le test de Ljung-Box est un test de blancheur. C'est un test statistique qui vise à rejeter ou non l'hypothèse H0
                 : Le résidu est un bruit blanc. Ici on lit sur la ligne Prob(Q) que la p-valeur de ce test est de 0.91
                , donc on ne rejette pas l'hypothèse.
                
                Le test de Jarque-Bera est un test de normalité. C'est un test statistique qui vise à rejeter ou non l'hypothèse H0
                 : Le résidu suit une distribution normale. Ici on lit sur la ligne Prob (JB) que la p-valeur du test est de 0
                . On rejette donc l'hypothèse.
                donc les résidus ne vérifient pas l'hypothèse de normalité
                
                Toutefois, lorsque nous analysons le plot residual errors, les résidus semblent tout de même suivre une distribution normale. Nous décidons donc de garder notre modèle.
                
                
                """)

    # Plot residual errors
    residuals = pd.DataFrame(s_model_fit.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    st.pyplot(fig)

    st.markdown("""
                Les résidus semblent tout de même suivre une distribution normale. Nous gardons notre modèle
                """)

    # représentation graphique des résultats grâce à la fonction plot_ crée 
    st.markdown("""
                Représentation graphique des prédictions :
                """)
    modele_sarima1 = plt.imread("modele_sarima1.jpg")
    st.image(modele_sarima1)
    
    st.markdown("""
                Nous avons représenté les prédictions et prévisions de ventes dans le futur à l’aide d’un graphique. 
                
                Nous avons distingué les données *train* des données *test* : 
                    
                - en rouge nous avons les données de ventes actuelles de l’échantillon train
                - en vert kaki nous avons les prédictions des ventes sur cette même période 
                - en vert foncé nous avons les données de ventes actuelles de l’échantillon test
                - en bleu, les prédictions de ventes de l’échantillon test
                - et enfin, en violet nous avons les prévisions de ventes de notre modèle sur le mois à venir

                Avec ce modèle, nous obtenons une erreur de prédiction de 42.34%. 
                
                Cette erreur a pu être améliorée en recherchant les paramètres optimaux du modèle.
                
                En effet, à l'aide de la fonction *auto_arima* du package pmdarima, nous avons pu déterminer les meilleurs paramètres pour notre modèle,
                i.e., ceux qui minimise la métrique AIC.
                
                Nous avons ainsi le modèle suivant :
                """)

    s_model = SARIMAX(endog=train_d , order=(0, 0, 1), seasonal_order=(0, 1, 2, 6), trend='t')
    s_model_fit=s_model.fit()
    st.write(s_model_fit.summary())

    st.markdown("""
                
                
                Là encore, l'hypothèse sur les bruits blancs est vérifiée mais non celle sur la normalité.
                Cependant, lorsque nous analysons le *Plot residual errors*, les résidus semblent tout de même suivre une distribution normale.
                
                Nous gardons notre modèle et nous obtenons les prédictions suivantes :
                """)

    modele_sarima2 = plt.imread("modele_sarima2.jpg")
    st.image(modele_sarima2)
    
    st.markdown("""
                Ce nouveau modèle nous permet d'obtenir une erreur de prédiction plus faible de **29.31%**.
                """)

if page == 'Conclusion':

    st.markdown("""
                Après avoir testé plusieurs modèles de régression non satisfaisants, 
                nous obtenons au final d'assez bons modèles de prédictions en utilisant 
                les algorithmes ARIMA et SARIMA.
                
                Notre étude nous a permis d'obtenir un modèle de prévision de vente 
                adapté à nos données et donnant ainsi une erreur de prédiction assez faible
                de 29%. Cette erreur de prédiction aurait pu être améliorée si nous avions 
                un historique plus large dans notre jeu de données (au moins 2 ans).
                En effet, il existe une saisonnalité d'une année à l'autre, 
                et cela nous aurait permis d'avoir un modèle plus performant.
                
                """)   
    