import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import get_dataframe
from machine_learning import lin_reg
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings

sns.set_theme()

st.set_page_config(page_title="Autoscout24 Interviewprojekt", layout="wide")

if "data" not in st.session_state:
    st.session_state.data, st.session_state.data_num, st.session_state.data_five = get_dataframe()

if "model" not in st.session_state:
    st.session_state.X, st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, \
        st.session_state.y_test, st.session_state.model = lin_reg(st.session_state.data)

pd.options.display.float_format = '{:.0f}'.format
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

col1, col2, col3 = st.columns([1, 5, 1])

with col2:
    st.title("Autoscout24")
    st.header("Dataframe")
    with st.expander("Drücken um den DataFrame anzuzeigen"):
        st.dataframe(st.session_state.data, width=1600)

    rows, columns = st.session_state.data.shape

    st.subheader("Über welchen Zeitraum wurden wie viele Autos verkauft?")
    yearly_counts = st.session_state.data['Jahr'].value_counts().sort_index()
    fig_all = plt.figure(figsize=(10, 6))
    bars = plt.bar(yearly_counts.index, yearly_counts.values, color='skyblue')
    plt.title('Anzahl der verkauften Autos pro jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl der verkauften Autos')
    plt.xticks(rotation=0)
    for bar, count in zip(bars, yearly_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha='center', va='bottom')

    st.pyplot(fig_all)
    st.markdown(f"Es wurden {rows} Autos im Zeitraum von {st.session_state.data['Jahr'].min()} bis "
                    f"{st.session_state.data['Jahr'].max()} verkauft")

    st.subheader("Welche Marken sind Erfasst?")

    unique_hersteller = st.session_state.data['Hersteller'].value_counts()
    filtered_hersteller = unique_hersteller[unique_hersteller > 100]
    fig_her = plt.figure(figsize=(10, 6))
    bars = plt.bar(filtered_hersteller.index, filtered_hersteller.values, color='skyblue')
    plt.title('Anzahl an verkauften Autos pro Hersteller (Y >= 100)')
    plt.xlabel('Hersteller')
    plt.ylabel('Anzahl an verkauften Autos')
    plt.xticks(rotation=90)
    for bar, count in zip(bars, filtered_hersteller.values):
        plt.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha='center', va='bottom', rotation=90)
    st.pyplot(fig_her)

    unique_hersteller = st.session_state.data['Hersteller'].sort_values().unique()
    st.caption(f"{unique_hersteller}")

    st.subheader("Korrelation zwischen den numerischen Features:")
    with st.expander("Drücken um einen Pairplot anzuzeigen!"):
        st.pyplot(sns.pairplot(data=st.session_state.data_num, diag_kind='hist'))

        st.caption("Erste Obversationen:")
        st.caption("Kilometerstand - Preis: Nicht wirklich korrelierend.")
        st.caption("Kilometerstand - PS: Je mehr PS das verkaufte Auto hat, desto kleiner ist der Kilometerstand der "
                   "verkauften Autos.")
        st.caption("Kilometerstand - Jahr: Je später das Auto verkauft wird, desto kleiner der Kilometerstand. ")
        st.caption("Preis - PS: Je mehr PS das Auto hat, desto höher ist der Preis. ")
        st.caption("Preis - Jahr: Je später das Auto verkauft wird, desto größer der Preis.")
        st.caption("PS - Jahr: Neuere Autos haben durchschnittlich mehr PS")

col1, col2 = st.columns([1, 1])

with col1:

    st.subheader("Observationen der gesamten numerischen Daten:")
    with st.expander("Drücken um nähere Analyse zu betrachten!"):
        st.pyplot(sns.lmplot(data=st.session_state.data, x='Kilometerstand', y='Preis', line_kws=dict(color="r"), fit_reg=True))
        st.pyplot(sns.lmplot(data=st.session_state.data, y='Kilometerstand', x='PS', line_kws=dict(color="r"), fit_reg=True))
        st.pyplot(sns.lmplot(data=st.session_state.data, y='Kilometerstand', x='Jahr', order=2, line_kws=dict(color="r"), fit_reg=True))
        st.pyplot(sns.lmplot(data=st.session_state.data, x='PS', y='Preis', line_kws=dict(color="r")))
        st.pyplot(sns.lmplot(data=st.session_state.data, x='Jahr', y='Preis', line_kws=dict(color="r")))
        st.pyplot(sns.lmplot(data=st.session_state.data, x='Jahr', y='PS', line_kws=dict(color="r")))

with col2:

    st.subheader("Observationen der numerischen Daten der Top 5 Hersteller:")
    with st.expander("Drücken um nähere Analyse zu betrachten!"):
        st.pyplot(sns.lmplot(data=st.session_state.data_five, x='Kilometerstand', y='Preis',
                             fit_reg=True, hue='Hersteller'))
        st.pyplot(
            sns.lmplot(data=st.session_state.data_five, y='Kilometerstand', x='PS', fit_reg=True, hue='Hersteller'))
        st.pyplot(
            sns.lmplot(data=st.session_state.data_five, y='Kilometerstand', x='Jahr', order=2,
                       fit_reg=True, hue='Hersteller'))
        st.pyplot(sns.lmplot(data=st.session_state.data_five, x='PS', y='Preis', hue='Hersteller'))
        st.pyplot(sns.lmplot(data=st.session_state.data_five, x='Jahr', y='Preis', hue='Hersteller'))
        st.pyplot(sns.lmplot(data=st.session_state.data_five, x='Jahr', y='PS', hue='Hersteller'))

col1, col2, col3 = st.columns([1, 5, 1])

with col2:

    st.subheader("Top 5 Autohersteller mit ihrem Durchschnittsverkaufspreis")
    sorted_data = st.session_state.data_five.groupby(['Hersteller']).Preis.mean().sort_values(ascending=False)
    for i, (manufacturer, mean_price) in enumerate(sorted_data.items(), start=1):
        st.markdown(f"{i}. {manufacturer}: {mean_price:.2f} €")
    st.pyplot(sns.lmplot(data=st.session_state.data_five, x='Kilometerstand', y='Preis',
                         fit_reg=True, hue='Hersteller', col='Hersteller'))
    st.pyplot(sns.lmplot(data=st.session_state.data_five, x='PS', y='Preis', hue='Hersteller', col='Hersteller'))
    st.pyplot(sns.lmplot(data=st.session_state.data_five, x='Jahr', y='Preis', hue='Hersteller', col='Hersteller'))


    st.subheader("Modell: Lineare Regression zur Berechnung des Preises")

    with st.expander("Drücken um das Vorhersage Modell anzuzeigen."):
        predictions = st.session_state.model.predict(st.session_state.X_test)
        slope, intercept = np.polyfit(st.session_state.y_test, predictions, 1)
        prediction_fig = plt.figure(figsize=(10, 4))
        plt.scatter(st.session_state.y_test, predictions, label='Daten Punkte')
        plt.xlabel("Korrekte Labels")
        plt.ylabel("Predictions")
        plt.plot(st.session_state.y_test, slope * st.session_state.y_test + intercept, color='red', label='Linear Regression Line')
        plt.legend()
        st.pyplot(prediction_fig)
        st.caption("Lineare Regression ist Teil des 'supervised machine learning'. Dabei lernt der Algorithmus wie er Input "
                   "Daten auf ein bestimmtest Ziel zuweist (hier der Preis). Lineare Regression wird zur Vorhersage von "
                   "Kontinuierlichen Zielen (hier der Preis) verwendet indem man die Beziehung zwischen den Inputs und den "
                   "Targets als lineare Gleichung darstellt. Oftmals wird das Modell genutzt um Regressions Analyse und "
                   "Predictive Modelling durchzuführen wo das Ziel ist basierend auf dem Input eine Vorhersage zu erstellen. "
                   "Ein Beispiel hierfür ist die Marktanalyse in der Wirtschaft.")

    st.subheader("Koeffizenz der Labels 'Kilometerstand', 'PS' und 'Jahr':")
    koeffizient = pd.DataFrame(st.session_state.model.coef_, st.session_state.X.columns)
    koeffizient.columns = ['Umsatz']
    st.dataframe(koeffizient)

    st.subheader("Fehlermetriken")
    MAE = metrics.mean_absolute_error(st.session_state.y_test, predictions)
    MSE = metrics.mean_squared_error(st.session_state.y_test, predictions)
    RSME = np.sqrt(metrics.mean_squared_error(st.session_state.y_test, predictions))
    st.write("MAE (Mean Absolute Error)", MAE)
    st.caption(f"Interpretation: Im Durschschnitt ist die Vorhersage {MAE} Einheiten vom echten Wert entfernt")
    st.write("MSE (Mean Squared Error)", MSE)
    st.caption(f"Interpretation: Im Durschschnitt sind die squared errors {MSE} Einheiten")
    st.write("RSME (Root Mean Squared Error)", RSME)
    st.caption(f"Interpretation: Im Durschschnitt ist die Vorhersage {RSME} Einheiten vom echten Wert entfernt")

    st.write("Je kleiner die Werte der Fehlermetriken sind, desto genauer ist das Modell")

    with st.expander("Drücken um die Fehler anzuzeigen"):
        error_fig = plt.figure(figsize=(10, 4))
        sns.histplot((st.session_state.y_test - predictions), kde=True)
        st.pyplot(error_fig)
        st.markdown("Je näher die Daten einer Normalverteilung ähneln, desto genauer ist das Ergebnis")

