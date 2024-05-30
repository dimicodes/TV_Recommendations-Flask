import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)


# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    ratings = db.Column(db.PickleType, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Load the dataset
dataset = pd.read_csv('IMDb Top TV Series.csv')

# Handle missing values
dataset.fillna('', inplace=True)

# Extract relevant columns
dataset = dataset[['Title', 'Year', 'Parental Rating', 'Rating', 'Number of Votes', 'Description']]

# Normalize numerical features
dataset['Rating'] = dataset['Rating'].astype(float)


# Function to convert 'Number of Votes' to an integer
def convert_votes(votes):
    if 'M' in votes:
        return int(float(votes.replace('M', '')) * 1_000_000)
    elif 'K' in votes:
        return int(float(votes.replace('K', '')) * 1_000)
    else:
        return int(votes)


# Apply the function to the 'Number of Votes' column
dataset['Number of Votes'] = dataset['Number of Votes'].apply(convert_votes)

# Remove leading numbers and any formatting from the 'Title' column
dataset['Title'] = dataset['Title'].str.replace(r'^\d+\.\s+', '', regex=True)

# Encode categorical features
dataset['Parental Rating'] = dataset['Parental Rating'].astype('category').cat.codes

# Converting descriptions into numerical vectors
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the description column
tfidf_matrix = tfidf.fit_transform(dataset['Description'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get the index of a TV series from its title
def get_index_from_title(title):
    return dataset[dataset['Title'] == title].index.values[0]


# Function to get the title of a TV series from its index
def get_title_from_index(index):
    return dataset.iloc[index]['Title']


# Function to recommend TV series
def recommend_tv_series(title, num_recommendations=5):
    # Get the index of the TV series that matches the title
    idx = get_index_from_title(title)

    # Get the pairwise similarity scores for all TV series
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the TV series based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar TV series
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the TV series titles
    tv_series_indices = [i[0] for i in sim_scores]
    recommendations = [get_title_from_index(i) for i in tv_series_indices]

    return recommendations


# Home route
@app.route('/')
def index():
    return redirect(url_for('login'))


# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        username = data['username']
        password = data['password']
        hashed_password = generate_password_hash(password, method='scrypt')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        username = data['username']
        password = data['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('rate'))
        return 'Invalid credentials'
    return render_template('login.html')


# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


# Rate TV series route
@app.route('/rate', methods=['GET', 'POST'])
@login_required
def rate():
    if request.method == 'POST':
        data = request.form
        title = data['title']
        rating = int(data['rating'])
        if current_user.ratings:
            ratings = current_user.ratings
        else:
            ratings = {}
        ratings[title] = rating
        current_user.ratings = ratings
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('rate.html', dataset=dataset)


# Recommend TV series route
@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    recommendations = []
    title = ""
    if request.method == 'POST':
        data = request.form
        title = data['title']
        num_recommendations = int(data['num_recommendations'])
        recommendations = recommend_tv_series_personalized(current_user, title, num_recommendations)
    return render_template('recommend.html', recommendations=recommendations, dataset=dataset, title=title)


def recommend_tv_series_personalized(user, title, num_recommendations=5):
    content_recommendations = recommend_tv_series(title, num_recommendations)
    if user.ratings:
        user_ratings = user.ratings
        rated_titles = list(user_ratings.keys())
        rated_indices = [get_index_from_title(t) for t in rated_titles]
        user_profile = np.mean(cosine_sim[rated_indices], axis=0)
        sim_scores = list(enumerate(user_profile))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if get_title_from_index(s[0]) not in rated_titles]
        personalized_recommendations = [get_title_from_index(s[0]) for s in sim_scores[:num_recommendations]]
        recommendations = list(set(content_recommendations + personalized_recommendations))[:num_recommendations]
    else:
        recommendations = content_recommendations
    return recommendations


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
