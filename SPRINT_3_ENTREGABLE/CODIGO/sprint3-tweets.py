import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import string
from nltk.corpus import stopwords

# Descargar el lexicon de VADER (solo la primera vez)
nltk.download('vader_lexicon')
nltk.download('stopwords')

def conectar_base_datos():
    """Conectar a la base de datos MySQL y obtener los tweets."""
    connection = pymysql.connect(
        host="127.0.0.1",
        user='root',
        password="",  
        database='tweets_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

def obtener_tweets(connection):
    """Obtener los textos y fechas de los tweets."""
    with connection.cursor() as cursor:
        query = "SELECT texto, fecha FROM tweets;"
        cursor.execute(query)
        return cursor.fetchall()

def limpiar_datos(tweets):
    """Limpiar datos y crear un DataFrame."""
    tweets_text = [tweet['texto'] for tweet in tweets]
    tweets_fecha = [tweet['fecha'] for tweet in tweets]
    
    df = pd.DataFrame({'texto': tweets_text, 'fecha': tweets_fecha})
    df = df.dropna(subset=['texto', 'fecha'])  # Eliminar filas con None
    df = df[df['texto'].str.strip() != '']  # Eliminar filas vacías
    return df

def analizar_sentimientos(df):
    """Aplicar análisis de sentimientos a los tweets."""
    sia = SentimentIntensityAnalyzer()
    df['sentimientos'] = df['texto'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['clase_sentimiento'] = df['sentimientos'].apply(
        lambda x: 'Positivo' if x > 0 else ('Negativo' if x < 0 else 'Neutral')
    )
    return df

def graficar_distribucion_sentimientos(df):
    """Graficar la distribución de sentimientos."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='clase_sentimiento', palette='viridis', hue='clase_sentimiento', legend=False)
    plt.title('Distribución de Sentimientos en Tweets')
    plt.xlabel('Clase de Sentimiento')
    plt.ylabel('Cantidad')
    plt.show()

    plt.figure(figsize=(8, 8))
    sentimiento_counts = df['clase_sentimiento'].value_counts()
    plt.pie(
        sentimiento_counts,
        labels=sentimiento_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['lightblue', 'lightcoral', 'lightgreen']
    )
    plt.title('Distribución de Sentimientos en Tweets')
    plt.axis('equal')  # Para que el gráfico de pastel sea un círculo
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='clase_sentimiento', y='sentimientos', data=df, palette='viridis')
    plt.title('Distribución de Puntajes de Sentimiento por Clase')
    plt.xlabel('Clase de Sentimiento')
    plt.ylabel('Puntuación de Sentimiento')
    plt.show()

def obtener_palabras_comunes(df):
    """Contar y visualizar las palabras más comunes en los tweets."""
    stop_words = set(stopwords.words('spanish'))

    def limpiar_texto(texto):
        texto = texto.lower()
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        palabras = texto.split()
        palabras = [palabra for palabra in palabras if palabra not in stop_words]
        return palabras

    df['tokens'] = df['texto'].apply(limpiar_texto)
    todas_palabras = [palabra for tokens in df['tokens'] for palabra in tokens]
    conteo_palabras = Counter(todas_palabras)

    palabras_comunes = conteo_palabras.most_common(10)
    print("Palabras más comunes:")
    for palabra, frecuencia in palabras_comunes:
        print(f"{palabra}: {frecuencia}")

    palabras, frecuencias = zip(*palabras_comunes)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(palabras), y=list(frecuencias), palette='viridis')
    plt.title('Palabras más comunes en los Tweets')
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()

def calcular_intensidad_promedio(df):
    """Calcular e imprimir la intensidad promedio de sentimientos."""
    intensidad_promedio = df.groupby('clase_sentimiento')['sentimientos'].mean()
    print("Intensidad promedio de cada sentimiento:")
    print(intensidad_promedio)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=intensidad_promedio.index, y=intensidad_promedio.values, hue=intensidad_promedio.index, palette='viridis', legend=False)
    plt.title('Intensidad Promedio de los Sentimientos en Tweets')
    plt.xlabel('Clase de Sentimiento')
    plt.ylabel('Intensidad Promedio')
    plt.show()

def clasificar_objetividad(df):
    """Clasificar los tweets en objetivos y subjetivos."""
    umbral_subjetividad = 0.5
    df['objetividad'] = df['sentimientos'].apply(
        lambda x: 'Objetivo' if -umbral_subjetividad <= x <= umbral_subjetividad else 'Subjetivo'
    )
    print(df[['texto', 'sentimientos', 'clase_sentimiento', 'objetividad']].head(10))

    conteo_objetividad = df['objetividad'].value_counts()
    print("\nConteo de Tweets Objetivos vs Subjetivos:")
    print(conteo_objetividad)

    plt.figure(figsize=(8, 6))
    sns.countplot(x='objetividad', data=df, hue='objetividad', palette='magma', legend=False)
    plt.title('Distribución de Tweets Objetivos vs Subjetivos')
    plt.xlabel('Tipo de Tweet')
    plt.ylabel('Cantidad')
    plt.show()

    intensidad_objetividad = df.groupby('objetividad')['sentimientos'].mean()
    print("\nIntensidad promedio de sentimientos en tweets objetivos y subjetivos:")
    print(intensidad_objetividad)

def main():
    connection = conectar_base_datos()
    try:
        tweets = obtener_tweets(connection)
        df = limpiar_datos(tweets)
        df = analizar_sentimientos(df)
        graficar_distribucion_sentimientos(df)
        obtener_palabras_comunes(df)
        calcular_intensidad_promedio(df)
        clasificar_objetividad(df)
    finally:
        connection.close()

if __name__ == "__main__":
    main()
