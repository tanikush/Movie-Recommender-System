from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans
import os 

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("tmdb_5000_movies.csv", low_memory=False)
movies = df[['title', 'genres', 'vote_average', 'popularity']].copy()
movies = movies.dropna(subset=['genres', 'vote_average', 'popularity'])

def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

movies['genre_list'] = movies['genres'].apply(parse_genres)

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies['genre_list'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(movies[['vote_average', 'popularity']])
scaled_df = pd.DataFrame(scaled_features, columns=['vote_average', 'popularity'])

final_features = pd.concat([genre_df, scaled_df], axis=1)

optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
movies['cluster'] = kmeans.fit_predict(final_features)

# Ensure 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')


# Movie recommendation function
# Movie recommendation function
def recommend_movies(movie_title, n=5):
    movie_title = movie_title.strip().lower()  # Ensure lowercase and remove any leading/trailing spaces
    target = movies[movies['title'].str.lower() == movie_title]

    if target.empty:
        return None  # If no movie found, return None

    cluster_id = target['cluster'].values[0]
    cluster_movies = movies[(movies['cluster'] == cluster_id) & 
                             (movies['title'].str.lower() != movie_title)]
    top_movies = cluster_movies[['title', 'vote_average', 'popularity']].sort_values(
        by=['vote_average', 'popularity'], ascending=False).head(n)
    
    return top_movies[['title']].values.flatten().tolist()  # Return as a simple list of movie titles



@app.route('/')
def index():
    return render_template('index.html', 
                           genre_plot='genre_distribution.png',
                           kmeans_plot='kmeans_diagnostics.png',
                           cluster_plot='cluster_counts.png')


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    recommendations = recommend_movies(movie_title)

    if recommendations is None or not recommendations:
        return render_template('recommend.html', error="No recommendations found for the movie title.")

    return render_template('recommend.html', recommendations=recommendations)



    # Generate a plot of genre distribution
def generate_genre_plot():
    genre_counts = movies['genre_list'].explode().value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
    plt.title('Genre Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Movies')
    plt.xlabel('Genre')
    plt.tight_layout()
    plt.savefig('static/genre_distribution.png')  # Save the plot as an image in the 'static' folder
    plt.close()

generate_genre_plot()  # Call the function to generate and save the plot

# Generate a plot of K-Means Clustering
def generate_kmeans_diagnostics():
    inertias = []
    silhouette_scores = []
    K_range = range(2, 21)

    from sklearn.metrics import silhouette_score

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(final_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(final_features, cluster_labels))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(K_range, inertias, 'o-', color='dodgerblue')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)

    ax2.plot(K_range, silhouette_scores, 's-', color='green')
    ax2.set_title('Silhouette Scores for Different k')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('static/kmeans_diagnostics.png')
    plt.close()

generate_kmeans_diagnostics()


# Generate a plot of Movie per Cluster

def generate_cluster_count_plot():
    cluster_counts = movies['cluster'].value_counts().sort_index()
    cluster_counts_df = cluster_counts.reset_index()
    cluster_counts_df.columns = ['cluster', 'count']

    plt.figure(figsize=(12, 5))
    sns.barplot(data=cluster_counts_df, x='cluster', y='count', hue='cluster', dodge=False, palette='coolwarm')
    plt.title('Number of Movies per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig('static/cluster_counts.png')
    plt.close()

generate_genre_plot()
generate_kmeans_diagnostics()
generate_cluster_count_plot()  


if __name__ == '__main__':
    app.run(debug=True)
