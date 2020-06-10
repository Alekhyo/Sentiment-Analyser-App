from flask import Flask, request,render_template,redirect, url_for
import requests,re


import numpy as np
import pandas as pd

from bs4 import BeautifulSoup





import pickle

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer





app=Flask(__name__)



word_dict=pickle.load(open('bow.pkl','rb'))
clb=pickle.load(open('model.pkl','rb'))


def remove_tags(df):
    """
        Removing HTML tags from text

    """
    df['review'] = df['review'].apply(lambda x: re.sub('<[^<]+?>', '', x))
    return df


def lower(df):
    """
        Converting to lower case

    """
    df['review'] = df['review'].str.lower()
    return df


def remove_punctuation(df):
    """
        Removing all punction marks and special characters

    """
    df['review'] = df['review'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x))
    return df


def remove_stopwords(df):
    """
        Remove stop words

    """
    words = np.array(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda x: [word for word in x.split() if word not in (words)])
    return df


def stemming(df):
    """
        Stemming

    """
    ss = SnowballStemmer('english')
    df['review'] = df['review'].apply(lambda x: ' '.join([ss.stem(i) for i in x]))
    return df

def prediction(word):
    sample=[]
    for i in word_dict:
        sample.append(word.split(" ").count(i[0]))

    return clb.predict(np.array(sample).reshape(1,1000))[0]

def scrape_info(id):

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'}


        req = requests.get("https://www.imdb.com/title/{}/".format(id), headers=headers)
        page = req.text
        soup = BeautifulSoup(page,"html.parser")

        movie_name = soup.find('div', {"class":"title_wrapper"}).find_all('h1')[0].text

        info = soup.find(class_='subtext').text.strip().replace('\n', '').strip().split('|')

        try:
            Duration = info[1].strip()
        except:
            Duration = "No Information Available"

        try:
            Genre = info[2].strip()
        except:
            Genre = "No Information Available"

        try:
            Release_date = info[3].strip()
        except:
            Release_date = "No Information Available"

        cast_crew = soup.find_all(class_='credit_summary_item')

        try:
            director = cast_crew[0].text.replace('\n', '')
        except:
            director = "No Information Available"

        try:
            actors = cast_crew[2].text.replace('\n', '').split('|')[0]
        except:
            actors = "No Information Available"

        try:
            imdb_rating = soup.find_all(class_='ratingValue')[0].text.replace('\n', '')
        except:
            imdb_rating = "No Rating"

        try:
            images = soup.find_all('img', src=True)
            image_src = [x['src'] for x in images]

            image_src = [x for x in image_src if x.endswith('.jpg')]
            poster = image_src[0]
        except:
            poster = "Poster Unavailable"

        review_page = requests.get("https://www.imdb.com/title/{}/reviews".format(id),headers=headers).text

        fetch_reviews = BeautifulSoup(review_page, 'html.parser')
        all_reviews = fetch_reviews.find_all(class_='review-container')

        reviews, review_date, reviewer_name, review_rating = [], [], [], []

        try:
            reviews = [i.find(class_="text show-more__control").text for i in all_reviews]
        except:
            reviews.append("No review")

        try:
            review_date = [i.find(class_='review-date').text for i in all_reviews]
        except:
            review_date.append("Information Unavailable")

        try:
            reviewer_name = [i.find(class_='display-name-link').text for i in all_reviews]
        except:
            reviewer_name.append("User Info Unavailable")

        for review in all_reviews:
            try:
                review_rating.append(int(review.find_all('span')[0].text.replace('\n', '').split('/')[0]))
            except:
                review_rating.append("Rating Unavailable")

        review_info = {'review': reviews, 'review_date': review_date, 'reviewer_name': reviewer_name,
                       'rating': review_rating}

        try:

            df = pd.DataFrame(review_info)
            df_aux=df.copy()

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=1000)


            df = remove_tags(df)
            df = lower(df)
            df = remove_punctuation(df)
            df = remove_stopwords(df)
            df = stemming(df)
            df['sentiment'] = df['review'].apply(prediction)
            df['sentiment'] = df['sentiment'].astype('int8')

            df.loc[(df['sentiment'] == 0) & (df['rating'] == 10), 'sentiment'] = 1
            df.loc[(df['sentiment'] == 0) & (df['rating'] == 9), 'sentiment'] = 1

            df['review'] = df_aux['review'].values
            sentiment_score = (df.loc[df['sentiment'] == 1].shape[0] / df.shape[0]) * 10
            sentiment_score=round(sentiment_score,1)

        except:
            df = pd.DataFrame()
            sentiment_score=0

        return movie_name, Duration, Genre, Release_date, director, actors, imdb_rating, poster, df,sentiment_score


def scrape_choice(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'}

    req = requests.get(url, headers=headers)
    page = req.text
    soup = BeautifulSoup(page, "html.parser")

    title_id, title, year = [], [], []

    pattern = r'[0-9]{4}'

    table = soup.find('table').find_all('tr')

    if len(table) < 10:
        for i in range(len(table)):
            title_id.append(str(table[i].find_all('a')[1]['href']).split('/')[-2])
            title.append(table[i].find_all('a')[1].text.strip())
            try:
                year.append(re.findall(pattern, table[i].find_all('td')[1].text)[0])
            except:
                year.append("No information available")
    else:
        for i in range(len(table[:11])):
            title_id.append(str(table[i].find_all('a')[1]['href']).split('/')[-2])
            title.append(table[i].find_all('a')[1].text.strip())
            try:
                year.append(re.findall(pattern, table[i].find_all('td')[1].text)[0])
            except:
                year.append("No information available")

    return title_id, title, year


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/choices', methods=['POST'])
def choices():
    title = request.form.get('Search')

    if len(title.split()) == 1:
        url = 'https://www.imdb.com/find?s=tt&q={}&ref_=nv_sr_sm'.format(title)
    else:
        title_str = "+".join(title.split())
        url = 'https://www.imdb.com/find?s=tt&q={}&ref_=nv_sr_sm'.format(title_str)

    title_id, movie_name, release_year = scrape_choice(url)

    return render_template('choices.html', title_id=title_id, movie_name=movie_name, release_year=release_year)


@app.route('/predict/<string:id>')
def predict(id):

        movie_name, Duration, Genre, Release_date, director, actors, imdb_rating, poster, df,sentiment_score = scrape_info(id)

        print(movie_name,Duration,Genre,Release_date,director,actors,imdb_rating,poster)

        return render_template('predict.html',movie_name=movie_name, duration=Duration, genre=Genre, release_date=Release_date, director=director,
                              actors=actors, imdb_rating=imdb_rating, poster=poster, df=df,sentiment_score=sentiment_score)



if __name__=="__main__":
    app.run(debug=True)