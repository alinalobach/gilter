from surprise import Dataset, Reader, KNNBasic
import pandas as pd

# Загрузка данных о рейтингах фильмов
ratings_data = pd.read_csv('ratings.csv')

# Создание объекта Reader для работы с данными о рейтингах
reader = Reader(rating_scale=(1, 5))

# Создание датасета Surprise на основе данных о рейтингах
data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)

# Использование алгоритма KNNBasic для построения модели коллаборативной фильтрации
model = KNNBasic()

# Обучение модели на данных
training_set = data.build_full_trainset()
model.fit(training_set)

# Функция для получения рекомендаций фильмов для пользователя
def get_recommendations(user_id, model, ratings_data, n=10):
    movie_ids = ratings_data['movieId'].unique()
    unrated_movies = [movie_id for movie_id in movie_ids if movie_id not in ratings_data[ratings_data['userId'] == user_id]['movieId']]
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movies]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    movie_indices = [int(recommendation.iid) for recommendation in recommendations]
    recommended_movies = ratings_data[ratings_data['movieId'].isin(movie_indices)]['title'].unique()
    return recommended_movies

# Пример использования
user_id = 1
recommendations = get_recommendations(user_id, model, ratings_data)
print(f"Рекомендации для пользователя с ID {user_id}:")
print(recommendations)
