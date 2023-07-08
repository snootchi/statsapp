import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn import linear_model
import math
from scipy.stats import chi2_contingency
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig
import plotly.express as px


d_tt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
alpha_tt = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
            2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086]
stdnt_table = pd.DataFrame(data={'ddl': d_tt, 'alpha': alpha_tt})

image = Image.open(r"C:\Users\JAKOB\Downloads\desktop-wallpaper-data-analytics-data-analysis.jpg")
st.image(image, use_column_width=True)
st.caption('Powered By : **StreamLit**')
st.title('Statistical Data Analysis')
st.markdown("""
This Webb Applicaion is dedicated to perform the various operations in **data analysis** as well as **statistical analysis**,
* This Web App was made by **RABIA MAHDI SALAH EDDINE** with the programming language **PYTHON**.
***
""")
data = st.file_uploader('Entrer data in format EXCEL (xlsx)')


if not data:
    st.info('The web are waiting for your File', icon="ℹ️")

else:
    df = pd.read_excel(data)
    df_cols = []

    for i in df.columns:
        df_cols.append(i)

    df_cols.append('aucun')
with st.sidebar:
    analyze = st.radio(
        "C'est quoi l'analyse que vous souhaitée ?",
        ('Analyse Bivariée', 'Analyse Multivariée')
    )
    if analyze == 'Analyse Bivariée':

        options = st.multiselect(
            "Selecter 2 colonnes/Variables pour y faire l'analyse :",
            df_cols)
        x = df[options[0]]
        y = df[options[1]]
        X = options[0]
        Y = options[1]

        st.write(
            "You've selected **{}** and **{}** Variables to Analyze em ".format(options[0], options[1]))
        test_name = st.selectbox("Quelle est l'analyse Bivariée que vous chercher :",
                                 ('aucun', 'KHI-DEUX TEST', 'Corrélation Linèaire', 'Regression Linéaire'))
    else:
        st.write("Il y l'analyse ACP dans l'analyse Multivariée")
        test_name = st.radio("Quelle est l'analyse Multivariée que vous chercher : ",
                             ('ACP', ''))

st.write("***")


def corr_lin(x, y):
    moy_x = sum(x)/len(y)
    moy_y = sum(y)/len(y)
    cov_xy = sum((a - moy_x)*(b - moy_y) for (a, b) in zip(x, y)) / len(x)
    etX = (sum((a - moy_x)**2 for a in x)/len(x))**0.5
    etY = (sum((b - moy_y)**2 for b in y)/len(y))**0.5
    r = round(cov_xy/(etX*etY), 3)
    matrix = df.corr()
    st.markdown('**La Matrice de La corrélation Linéaire**')
    st.dataframe(matrix.style.highlight_max(axis=0))
    st.success('**Coefficient de Corrélation Souhaitées** {}'.format(r))
    # Set up the matplotlib plot configuration
    #
    f, ax = plt.subplots(figsize=(12, 6))
    #
    # Generate a mask for upper traingle
    #
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    #
    # Configure a custom diverging colormap
    #
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    #
    #    Draw the heatmap
    #
    st.write("**Matrice de Corrélation avec HEATMAP**")
    sns.heatmap(matrix, annot=True, mask=mask, cmap=cmap)
    f
    # Tester la significativité de la corrélation linéaire
    sr_value = []
    t_value = []
    for i in matrix.values.reshape(-1):
        if i == 1:
            pass
        else:
            sr_value.append(math.sqrt((1-i**2)/10))
            t_value.append(((abs(i))/math.sqrt((1-i**2)/10)))

    data_test_t = {'Sr': sr_value, 'T_Valeurs': t_value}

    df_st_t = pd.DataFrame(data_test_t)
    var_c = []
    for i in matrix.index:
        for j in matrix.index:
            if i == j:
                pass
            else:
                var_c.append((i, j))

    df_st_t['variables_de_liaisons'] = var_c
    ddl = len(x)-2
    for i in stdnt_table.values:
        if i[0] == ddl:
            t_seuil = i[1]

    st.write("On a la valeur de **t_seuil = {}**".format(t_seuil))

    interp = []
    for i in df_st_t['T_Valeurs']:
        if abs(i) >= t_seuil:
            interp.append('Rejet H0 , liaison linéaire significatif')
        else:
            interp.append('Non rejet de H0 , pas de liaison linéaire')
    df_st_t['Interpretation'] = interp
    df_st_t.drop_duplicates(subset=['Sr'], keep='first', inplace=True)

    st.write('**Tester La significativité des Corrélations Linéaires**')
    st.table(df_st_t)


def linear(x, y):
    st.subheader('Répresentation graphique Linéaire')

    f = plt.figure(figsize=(12, 6))
    plt.scatter(x=X, y=Y, data=df, marker='+', color='red')
    sns.regplot(x=X, y=Y, data=df, color='red')
    plt.xlabel('{}'.format(options[0]))
    plt.ylabel('{}'.format(options[1]))
    plt.title('Le Model Linéaire entre le Variable {} et le variable {}'.format(
        options[0], options[1]))

    plt.show()
    f
    reg = linear_model.LinearRegression()
    x_x = df[[X]]
    y_y = df[Y]
    reg.fit(x_x, y_y)

    st.success('''
    le **coefficient de regression** = {} , et **lintercept** = {}'''.format(reg.coef_, reg.intercept_))
    st.subheader("L'équation de regression linéaire est donc :")
    st.success("**{}** = **{} x {} + ({})**".format(Y,
               np.round_(reg.coef_, 3), X, reg.intercept_))


def kh2(x, y):
    # Contruire le tableau de contingence
    cont = df[[X, Y]].pivot_table(
        index=X, columns=Y, aggfunc=len).fillna(0).copy()

    tx = df[X].value_counts()
    ty = df[Y].value_counts()
    cont = cont.astype(int)
    st.subheader('Tableau de Contingence : ')
    st.table(cont)
    # Construire le Tableaux des effectifs Théoriques
    tx_df = pd.DataFrame(tx)
    tx_df.columns = ["c"]
    ty_df = pd.DataFrame(ty)
    ty_df.columns = ["c"]

    # Valeurs totaleso observées
    n = len(df)

    # Produit matriciel. On utilise pd.T pour pivoter une des deux séries.
    effth = (tx_df.dot(ty_df.T) / n)
    st.subheader('Tableau des effectifs theoriques :')
    st.dataframe(effth.style.highlight_max(axis=0))

    # Calculer la statistique Khi2 et la p-valeur
    st_chi2, p_val, ddl, st_exp = chi2_contingency(cont)
    st.latex(r'''\Huge \chi^{2}''')
    st.success("**La Statistique KH2** = {}".format(st_chi2))
    st.success("**La P-VALEUR** = {}".format(p_val))
    st.write("Avec un **DDL** = {} et **ALPHA** = {}".format(ddl, 0.05))
    if p_val > 0.05:
        st.latex(r'''Pour \ que\ on \ a \ P-Valeur \ > \alpha(0.05)\ Donc\ on \  rejet \ H1 \ et\ on\ reste \ sur \ H0 \ ou\ il\ y\ a\ une\ Indépendance''')
    else:
        st.latex(r'''Pour \ que\ on\ a\ P-Valeur \ <\ \alpha(0.05)\ Donc\ on\ Accept\ H1 \ et \ Rejette\ H0\ et \ qui \ montre \ une \Dépendance''')


def ACP(data, n_components):
    mypca = PCA()
    # Transformer les valeurs on loi normale
    ss = StandardScaler()
    ss.fit(data)
    scaled_df = ss.transform(data)
    scaled_df = pd.DataFrame(scaled_df, columns=data.columns)
    mypca.fit_transform(scaled_df)
    # Schématiser la covariance
    st.subheader("Matrice des covariances")
    cov = scaled_df.cov()
    st.dataframe(cov.style.highlight_max(axis=0))
    # EigenStuff
    st.subheader("EigenStuff")
    eigenvalues, eigenvectors = eig(cov)
    st.latex(r"Valeures Propres \  \lambda i")
    ssss = mypca.explained_variance_
    eig_cols = [f'F{i}' for i in list(range(1, len(data.columns)+1))]
    va_p = pd.DataFrame(data=ssss, columns=['valeurs_propres'], index=eig_cols)
    va_p['Variabilité (%)'] = (mypca.explained_variance_ratio_)*100
    va_p[" (%) cumulé"] = va_p.loc[:, 'Variabilité (%)'].cumsum()
    st.table(va_p.T)
    st.latex(r"Vecteurs Propres ")
    ve_p = pd.DataFrame(data=eigenvectors,
                        columns=eig_cols, index=data.columns)
    st.table(ve_p)

    # Modélisation
    pca = PCA(n_components=n_components)
    pca.fit(scaled_df)
    x_pca = pca.transform(scaled_df)
    pca_df = pd.DataFrame(data=x_pca)
    st.subheader("Tableau des Pricipaux components")
    st.table(pca_df)


st.header("Effectuation de l'Analyse souhaitée")
df_cols_cpp = df_cols.copy()
df_cols_cpu = df_cols.copy()
if test_name == 'aucun':
    st.warning('Waiting for your Option to Analyze what you want ', icon="⚠️")
else:
    st.subheader(test_name)

if test_name == 'KHI-DEUX TEST':
    kh2(x, y)

if test_name == 'Corrélation Linèaire':
    corr_lin(x, y)


if test_name == 'Regression Linéaire':
    linear(x, y)


@st.experimental_memo
def plott(data, features):
    fig = px.scatter_matrix(data, dimensions=features, height=800, width=800)
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig)


@st.experimental_memo
def plot_tg(data, features, target):
    fig = px.scatter_matrix(data, dimensions=features, color="{}".format(
        target), height=800, width=800)
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig)


def plot_pca(data):
    pca = PCA()
    ft = []
    for i in data.columns:
        ft.append(i)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(data=scaled, columns=ft)
    components = pca.fit_transform(scaled_df[ft])
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(4),
        height=800, width=800
    )
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig)


def plot_cercle(data):
    # Visualiser le cercle des correlations
    pca = PCA()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    features_names = []
    for i in data.columns:
        features_names.append(i)
    scaled_df = pd.DataFrame(data=scaled, columns=features_names)

    components = pca.fit_transform(scaled_df[features_names])
    loadings = pca.components_
    n_features = pca.n_features_

    pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
    # Match PC names to loadings
    pc_loadings = dict(zip(pc_list, loadings))
    # Matrix of corr coefs between feature names and PCs
    loadings_df = pd.DataFrame.from_dict(pc_loadings)
    loadings_df['Variables'] = features_names
    loadings_df = loadings_df.set_index('Variables')
    # Get the loadings of x and y axes
    xs = loadings[0]
    ys = loadings[1]

    # Plot the loadings on a scatterplot
    fig = plt.figure(figsize=(15, 10))
    for i, varnames in enumerate(features_names):
        plt.scatter(xs[i], ys[i], s=200)
        plt.arrow(
            0, 0,  # coordinates of arrow base
            xs[i],  # length of the arrow along x
            ys[i],  # length of the arrow along y
            color='r',
            head_width=0.01
        )
        plt.text(xs[i], ys[i], varnames, fontsize=16)

        # Define the axes
    xticks = np.linspace(-1, 1, num=5)
    yticks = np.linspace(-1, 1, num=5)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel('F1 ({} %)'.format(
        np.round(pca.explained_variance_ratio_[0]*100)), fontsize=15)
    plt.ylabel('F2 ({} %)'.format(
        np.round(pca.explained_variance_ratio_[1]*100)), fontsize=15)

    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    # Show plot
    plt.title("Cercle des Corrélations avec ({}) {} d'informations".format(np.round(
        pca.explained_variance_ratio_[0]*100) + np.round(pca.explained_variance_ratio_[1]*100), '%'), fontsize=20)
    st.pyplot(fig)


if test_name == 'ACP':
    target_var = st.radio(
        "Choisir un 'Target-Variable' si trouvé",
        df_cols)
    st.subheader("Visualisation des variables")
    if target_var == 'aucun':
        df_cols_cpu.remove('aucun')
        plott(df, df_cols_cpu)
        df_cols_cpu.append('aucun')
    else:
        df_cols_cpp.remove('aucun')
        plot_tg(df, df_cols_cpp, target_var)
        df_cols_cpp.append('aucun')

    st.subheader("Niveau/Nombre des composantes PCi")
    y = st.slider('Choisir Nombre des composantes souhaitées :',
                  min_value=1, max_value=3, step=1)
    ACP(df, y)
    st.subheader("Les principaux composantes avec (%) des informations ")
    plot_pca(df)
    st.subheader("Visualisation le Cercle des corrélations")
    plot_cercle(df)

st.experimental_memo.clear()
