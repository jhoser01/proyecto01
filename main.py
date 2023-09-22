from fastapi import FastAPI
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#iniciamos FastAPI y el dataframe
app = FastAPI()

game = pd.read_csv("G:/Mi unidad/SOYHENRY/CIENCIA DE DATOS/CURSO/PROYECTOS/PI1_ML 2.0/DATA_LIMPIA/gameclean.csv")
item = pd.read_csv("G:/Mi unidad/SOYHENRY/CIENCIA DE DATOS/CURSO/PROYECTOS/PI1_ML 2.0/DATA_LIMPIA/itemclean.csv")
review = pd.read_csv("G:/Mi unidad/SOYHENRY/CIENCIA DE DATOS/CURSO/PROYECTOS/PI1_ML 2.0/DATA_LIMPIA/reviewclean.csv")

#definimos la función que transformará en lista todos aquellos datos que perdieron esta condición
def obtener(celda):
    if pd.isnull(celda):
        return None
    if isinstance(celda, str) and celda.startswith("[") and celda.endswith("]"):
        try:
            return ast.literal_eval(celda)
        except (ValueError, SyntaxError):
            return celda  # Devuelve la celda original si no se puede convertir a lista
    return celda

item = item.applymap(obtener)
game = game.applymap(obtener)
review = review.applymap(obtener)

# Expandir las listas en las columnas 'item_id' y 'id'
item_df = item.explode('item_id')

game["id"] = game["id"].astype(str)

# Unir los DataFrames en base a 'item_id' e 'id'
funone = pd.merge(item_df, game, left_on='item_id', right_on='id')

#Restaurar la estructura de lista en 'item_id' y 'id'
funone = funone.groupby('user_id').agg({'item_id': list, 'price': list}).reset_index()


@app.get("/userdata/{user_id}")
def userdata(user_id: str):
    user_data = funone[funone['user_id'] == user_id]

    total = 0
    
    if user_data.empty:
        return "No se encontró el usuario"  # Devuelve si el usuario no se encuentra
    
    if user_data.empty:
        total = 0   # Si no se encuentra el usuario, devolvemos 0 como suma
    
    for precio in user_data['price'].iloc[0]:
        try:
            total += float(precio)
        except ValueError:
            pass

    
    
    recommend_list = user_data['recommend'].iloc[0]
    if not recommend_list:
        porcentage = 0  # Si la lista está vacía, el porcentaje es 0
    percentage = (sum(recommend_list) / len(recommend_list)) * 100

   
    id_item_list = user_data['item_id'].iloc[0]
    if not id_item_list:
        return 0  # Si la lista está vacía, el conteo es 0
    cantidad_items = sum(1 for item in id_item_list if item is not None)

    return {'Cantidad de dinero gastado': total, 'porcentaje de recomendación':porcentage, 'cantidad de items':cantidad_items}


@app.get("/recomendacion_juego/{product_id}")
async def recomendacion_juego(product_id: int):
    try:
        # Obtener el ID del juego
        target_game = game[game['id'] == product_id]

        if target_game.empty:
            return {"message": "No se encontró el juego de referencia."}

        # Combina las etiquetas (tags) y géneros en una sola cadena de texto
        target_game_tags_and_genres = ' '.join(target_game['tags'].fillna('').astype(str) + ' ' + target_game['genres'].fillna('').astype(str))

        # Crea un vectorizador TF-IDF
        tfidf_vectorizer = TfidfVectorizer()

        # Configura el tamaño del lote para la lectura de juegos
        chunk_size = 100  # Tamaño del lote (puedes ajustarlo según tus necesidades)
        similarity_scores = None

        # Procesa los juegos por lotes utilizando chunks
        for chunk in pd.read_csv('G:/Mi unidad/SOYHENRY/CIENCIA DE DATOS/CURSO/PROYECTOS/PI1_ML 2.0/DATA_LIMPIA/gameclean.csv', chunksize=chunk_size):
            # Combina las etiquetas (tags) y géneros de los juegos en una sola cadena de texto
            chunk_tags_and_genres = ' '.join(chunk['tags'].fillna('').astype(str) + ' ' + chunk['genres'].fillna('').astype(str))

            # Aplica el vectorizador TF-IDF al lote actual de juegos y al juego de referencia
            tfidf_matrix = tfidf_vectorizer.fit_transform([target_game_tags_and_genres, chunk_tags_and_genres])

            # Calcula la similitud entre el juego de referencia y los juegos del lote actual
            if similarity_scores is None:
                similarity_matrix = cosine_similarity(tfidf_matrix)
                similarity_scores = cosine_similarity(similarity_matrix, similarity_scores)
            else:
                similarity_matrix = cosine_similarity(tfidf_matrix)
                similarity_scores = cosine_similarity(similarity_matrix, similarity_scores)

        if similarity_scores is not None:
            # Obtiene los índices de los juegos más similares
            similar_games_indices = similarity_scores[0].argsort()[::-1]

            # Recomienda los juegos más similares (puedes ajustar el número de recomendaciones)
            num_recommendations = 5
            recommended_games = game.loc[similar_games_indices[1:num_recommendations + 1]]

            # Devuelve la lista de juegos recomendados
            return recommended_games[['app_name','id']].to_dict(orient='records')

        return {"message": "No se encontraron juegos similares."}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}