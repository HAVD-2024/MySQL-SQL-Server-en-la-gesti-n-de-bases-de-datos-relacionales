import pandas as pd
import pymysql

# Ruta del archivo CSV
file_path = r'C:\Users\myvlad\Desktop\Udemy courses\FreeLance\tweets_extraction.csv'

# Leer el archivo CSV
df = pd.read_csv(file_path)

# Reemplazar NaN por None
df = df.where(pd.notnull(df), None)

# Conectar a la base de datos MySQL
connection = pymysql.connect(
    host="127.0.0.1",
    user='root',
    password="",  # Asegúrate de ingresar tu contraseña si la tienes
    database='tweets_db',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        # Insertar los datos de CSV en la tabla 'tweets'
        insert_query = """
        INSERT INTO tweets (id_tweet, usuario, texto, fecha, retweets, favoritos, hashtags)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        for _, row in df.iterrows():
            cursor.execute(insert_query, (
                row['ID'],               # id_tweet del tweet
                row['Usuario'], 
                row['Texto'], 
                row['Fecha'], 
                row['Retweets'], 
                row['Favoritos'], 
                row['Hashtags']
            ))
    
    # Confirmar los cambios en la base de datos
    connection.commit()
    print("Datos insertados correctamente en la base de datos.")

except pymysql.MySQLError as e:
    print(f"Error al insertar los datos: {e}")
    connection.rollback()

finally:
    # Cerrar la conexión a la base de datos
    connection.close()
