from __future__ import print_function
from flask import Flask, render_template, flash, url_for, request, redirect, session, json, abort
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from sqlalchemy.orm import sessionmaker
import redis
from sqlalchemy import create_engine, join
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import re
import psycopg2
from datetime import datetime, timedelta
import sys
import threading

# from flask_login import  login_required
# from flask_login import UserMixin, login_user, LoginManager, current_user, logout_user
# from flask_bcrypt import Bcrypt


app = Flask(__name__)
# bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:albertsani10@localhost/skripsi'
db = SQLAlchemy()
db.init_app(app)
# Configure Flask-Session to use Redis as the session interface
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
Session(app)
# conn = psycopg2.connect(database="skripsi", user="postgres", password="albertsani10", host="localhost", port="5432")


def connect_db():
    conn = psycopg2.connect(app.config['SQLALCHEMY_DATABASE_URI'])
    return conn


class Collections(db.Model):
    __tablename__ = "collections"
    knokat = db.Column(db.String)
    kode_buku = db.Column(db.String, primary_key=True)
    title = db.Column(db.String)
    author = db.Column(db.String)
    publisher = db.Column(db.String)
    subject = db.Column(db.String)
    city = db.Column(db.String)
    year = db.Column(db.String)
    language = db.Column(db.String)
    mata_kuliah = db.Column(db.String)
    jurusan = db.Column(db.String)
    # circulations = db.relationship('Circulations', backref='collections')


class Users(db.Model):
    __tablename__ = "users"
    nomor_induk = db.Column(db.String, primary_key=True)
    username = db.Column(db.String)
    password = db.Column(db.String)
    # circulations = db.relationship('Circulations', backref='users')


# class Circulations(db.Model):
#     __tablename__ = "circulations"
#     trans_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     nomor_induk = db.Column(db.String, db.ForeignKey('users.nomor_induk'))
#     kode_buku = db.Column(db.String, db.ForeignKey('collections.kode_buku'))
#     tanggal_pesan = db.Column(db.DateTime)
#     tanggal_pinjam = db.Column(db.DateTime)
#     tanggal_batas = db.Column(db.DateTime)
#     tanggal_kembali = db.Column(db.DateTime)
#     status_pinjam = db.Column(db.Integer)  # 1x
#     status_terlambat = db.Column(db.Integer)  # 1x
#     bst_bayar = db.Column(db.String)  # Y
#     tanggal_bayar = db.Column(db.DateTime)
#     no_kw = db.Column(db.Integer)  # 3424
#     trans_status = db.Column(db.String)
#     trans_tipe = db.Column(db.String)
#     trans_id_ref = db.Column(db.Integer)
#     kode_operator = db.Column(db.String)
#     rating = db.Column(db.Integer)
#     review = db.Column(db.String)
#     tanggal_update = db.Column(db.DateTime)
#     # collections = db.relationship('Collections', back_populates="circulations")
#     # users = db.relationship('Users', back_populates="circulations")


class Ratings(db.Model):
    __tablename__ = "ratings"
    nomor_induk = db.Column(db.String)
    knokat = db.Column(db.String)
    kode_buku = db.Column(db.String, primary_key=True)
    rating = db.Column(db.Integer)

# # route stuff


@app.route('/', methods=['POST', 'GET'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        nrp = request.form['nrp']
        password = request.form['password']
        print(request.form)  # print the form values
        print(nrp)

        user = Users.query.filter_by(
            nomor_induk=nrp, password=password).first()
        print(user)
        if user:
            session['nomor_induk'] = nrp  # input no induk ke session
            session['username'] = user.username  # input username ke session
            return redirect(url_for('home'))
        else:
            return render_template('login.html')

    return render_template('login.html')


@app.route('/home', methods=['POST', 'GET'])
# @login_required
def home():
    mahasiswa_id = session.get('nomor_induk')
    if mahasiswa_id is None:
        return redirect(url_for('login'))
    else:
        username = session.get('username')  # ambil username dr session
        # Fetch user data from database and render dashboard template
        user = db.session.query(Users).filter_by(
            nomor_induk=mahasiswa_id).first()
        engine = create_engine(
            'postgresql+psycopg2://postgres:albertsani10@localhost:5432/skripsi')
        Session = sessionmaker(bind=engine)
        books_df = pd.read_sql_query('SELECT * FROM collections', engine)
        full_books_df = pd.read_sql_query('SELECT * FROM collections', engine)
        print(books_df)
        # rename nama kolom
        full_books_df.rename(columns={'kode_buku': 'book_id'}, inplace=True)
        full_books_df['book_id'] = full_books_df['book_id'].apply(
            lambda x: x.strip())
        books_df['subject'] = books_df['subject'].fillna(
            books_df['title'])  # fillna subject dengan title

        # subject preprocessing
        # mengupper casekan semua subject agar perhitungan pada vector tidak salah
        books_df['subject'] = books_df['subject'].str.upper()
        clean_special_char = re.compile('[/(){}\[\]\|@;-]')
        # lambda function digunakan untuk mempassing setiap element dari kolom subject sebagai argument ke sub()
        books_df['subject'] = books_df['subject'].apply(
            lambda x: clean_special_char.sub(' ', x))
        # replace/subtract subject yg memiliki karakter seperti pada clean_special_char

        # pengganti rbooks
        books_df.drop(['knokat', 'author', 'publisher', 'city', 'year', 'language',
                    'mata_kuliah'], inplace=True, axis=1)  # menghilangkan tabel yang tidak diperlukan
        books_df.rename(columns={'kode_buku': 'book_id'},
                        inplace=True)  # rename nama kolom
        # menghilangkan data yang duplicates berdasarkan column book_id
        books_df = books_df.drop_duplicates(subset='book_id')
        books_df = books_df.sort_values(by='book_id', ascending=True).reset_index(
            drop=True)  # mengurutkan kode buku dan mereset indexnya
        books_df['book_id'] = books_df['book_id'].apply(
            lambda x: x.strip())  # menghilangkan spasi dari data didatabase
        # menghilangkan spasi dari data didatabase
        books_df['title'] = books_df['title'].apply(lambda x: x.strip())
        # print(books_df)

        ratings_df = pd.read_sql_query('SELECT * FROM ratings', engine)
        # ratings_df head()
        # menghilangkan tabel yang tidak diperlukan
        ratings_df.drop('knokat', inplace=True, axis=1)
        ratings_df.rename(
            columns={'nomor_induk': 'mahasiswa_id', 'kode_buku': 'book_id'}, inplace=True)
        # rat = ratings_df groupby('nomor_induk').kode_buku.apply(','.join).reset_index()
        ratings_df['mahasiswa_id'] = ratings_df['mahasiswa_id'].apply(
            lambda x: x.strip())  # menghilangkan spasi dari data didatabase
        ratings_df['book_id'] = ratings_df['book_id'].apply(
            lambda x: x.strip())  # menghilangkan spasi dari data didatabase
        # print(ratings_df)
        if mahasiswa_id not in ratings_df['mahasiswa_id'].values:
            return render_template('home2.html',head="SISTEM REKOMENDASI BUKU PERPUSTAKAAN")
        else:
            # user sedang login
            selected_mahasiswa_id = session.get('nomor_induk')

            # diambil berdasarkan semester buku yang pernah dibaca
            # Filter ratings_df to only include rows with the selected_mahasiswa_id
            full_books_df['year'] = full_books_df['year'].fillna(0).apply(int)
            merged_df_all = pd.merge(ratings_df, books_df, on='book_id')
            user_ratings_df_all = merged_df_all[merged_df_all['mahasiswa_id'] ==
                                                selected_mahasiswa_id].sort_values(by='semester', ascending=True)

            merged_df_all_book = pd.merge(ratings_df, full_books_df, on='book_id')
            user_ratings_df_all_book = merged_df_all_book[merged_df_all_book['mahasiswa_id'] ==
                                                selected_mahasiswa_id].sort_values(by='semester', ascending=True)
            # Extract the book IDs from user_ratings_df
            book_ids_all = user_ratings_df_all['book_id'].tolist()

            # mengambil 5 buku untuk test
            # num_samples = len(ratings_df[ratings_df['mahasiswa_id'] == selected_mahaiswa_id]['book_id'].values)
            # test_size = 5/num_samples

        # mengsplit data menjadi 90 train : 10 test
        # Split the list into training and testing sets
            user_books_train, user_books_test = train_test_split(
                book_ids_all, test_size=5, shuffle=False)

            # content-based filtering

            def contentBased(user_train_data, books_df):
                # merge all_rated_train(user_rated_train) book_id dengan subjectnya
                merge_rated_book_subject = pd.merge(
                    pd.Series(user_train_data, name='book_id'), books_df, on='book_id')
                # print(merge_rated_book_subject)

                # merge all_rated_book_subject
                # menyatukann semua subject dari all_rated_train selected_user
                all_rated_book_subject = ','.join(merge_rated_book_subject.subject)
                # print(all_rated_book_subject)
                all_rated_book_subject_df = pd.DataFrame(
                    {'subject': all_rated_book_subject}, index=[0])  # dimasukkan ke dalam dataframe
                # print(all_rated_book_subject_df)
                all_rated_book_subject_df['subject'] = all_rated_book_subject_df['subject'].str.lower(
                )
                # print(all_rated_book_subject_df)

                # tf-idf
                tfidf = TfidfVectorizer(
                    stop_words='english',  analyzer='word', ngram_range=(1, 6))
                # (khusus untuk data training)untuk digunakan pada data pelatihan sehingga kita dapat menskalakan data pelatihan dan juga mempelajari parameter penskalaan dari data tersebut.
                all_book = tfidf.fit_transform(books_df['subject'])
                # print(tfidf.get_feature_names_out())
                # print('@@@@@@@')
                # print(tfidf.vocabulary_)
                # print(all_book.shape)
                # print(all_book)
                # print('all rated:',all_rated_book_subject_df['subject'])
                # Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data.
                user_rated_train_transform = tfidf.transform(
                    all_rated_book_subject_df['subject'])

                # cosine similarity
                sim = cosine_similarity(user_rated_train_transform, all_book)
                # print(sim.shape)
                sim = sim.transpose()  # transpose matrix agar berbentuk vertikal column sehingga dapat
                # print(sim)
                similarity_df = pd.DataFrame(columns=['book_id', 'similarity'])
                for i in range(len(books_df)):
                    similarity_df = pd.concat([similarity_df, pd.DataFrame(
                        {'book_id': [books_df['book_id'][i]], 'similarity': [sim[i]]})], ignore_index=True)
                similarity_df = similarity_df.sort_values(
                    by=['similarity'], ascending=False)
                # print(similarity_df)
                # print(user_train_data)
                # print(user_books_test)
                # drop similarity data based on user train and test data
                similarity_df = similarity_df[~similarity_df['book_id'].isin(
                    user_train_data)]
                # The tilde (~) operator is used to negate the boolean values returned by the isin() method

                # sort values
                similarity_df = similarity_df.sort_values('similarity', ascending=False).reset_index(
                    drop=True)  # sort data dari nilai similarity terbesar
                # merged_df = pd.merge(similarity_df, books_df[['book_id','subject', 'title','jurusan']], on='book_id', how='left')

                return similarity_df

            # USER COLLABORATIVE FILTERING FUNCTION FIX
            # coba mean
            def userCollaborativeFiltering(mahasiswa_id, ratings_data_df, user_train_data, books_data_df, k):
                # vector ganti
                column_name = 'book_id'
                index_name = 'mahasiswa_id'
                values_name = 'rating'

                # Create pivot table for book rating
                pivot_rating = pd.pivot_table(
                    ratings_data_df, values=values_name, index=index_name, columns=column_name)
                # Fill missing values with the mean value of each user
                # Note coba fillna diganti dengan median, 3 (nilai netral)
                user_mean = pivot_rating.median(axis=1)
                # print('every user mean')
                # print(user_mean)
                # fill na with pivot_rating mean per user, because we want to get the mean rating per user
                pivot_rating = pivot_rating.apply(
                    lambda mean: mean.fillna(user_mean[mean.name]), axis=1)
                # pivot_rating = pivot_rating.fillna(0)
                # print(pivot_rating)

                # Get user's book rating untuk menemukan cosine similarity
                user_rating = pivot_rating.loc[mahasiswa_id]
                # print('user_rating')
                # print(user_rating)
                # print(user_rating.shape)

                # Drop selected user from pivot_rating, menghilangkan user dari pivot table
                others_rating = pivot_rating.drop(index=[mahasiswa_id])
                # print('others_rating')
                # print(others_rating)

                # Create dataframe to contain similarity results
                similarity_result_df = pd.DataFrame(
                    columns=['selected_mahasiswa_id', 'other_mahasiswa_id', 'similarity'])

                # Iterate through others_rating
                for other_user, mean in others_rating.iterrows():
                    # print([others_rating.loc[other_user]])
                    # mencari similarity antara semua buku yg telah dibaca user(train) dengan seluruh data rating pengguna lainnya
                    sim = cosine_similarity(
                        [user_rating], [others_rating.loc[other_user]])
                    similarity_result_df = pd.concat([similarity_result_df, pd.DataFrame({'selected_mahasiswa_id': mahasiswa_id, 'other_mahasiswa_id': [
                                                    other_user], 'similarity': [sim]})])  # masukkan hasil similarity pada dataframe baru
                similarity_result_df = similarity_result_df.sort_values(
                    by='similarity', ascending=False)
                # print(similarity_result_df)
                # print(similarity_result_df.shape)

                # Sort data by similarity and get k data
                similar_users = similarity_result_df.sort_values(
                    'similarity', ascending=False).reset_index(drop=True)
                # pakai .head(k) karena data pada dataframe hasil concat
                similar_users_k = similar_users.head(k)
                # print('k similar_users')
                # print(similar_users_k)

                # Get book_id and rating by mahasiswa_id from similar_users
                similar_users_data = pd.DataFrame(
                    columns=[index_name, column_name, values_name])
                for i in range(len(similar_users_k.index)):
                    similar_users_data = pd.concat(
                        [similar_users_data, ratings_data_df.loc[ratings_data_df['mahasiswa_id'] == similar_users_k.at[i, 'other_mahasiswa_id']]])
                # print('similar_users_data')
                # print(similar_users_data)
                # print(similar_users_data.shape) #2553

                # Drop books rated by selected user (train+test)
                # The tilde (~) operator is used to negate the boolean values returned by the isin() method
                similar_users_clear = similar_users_data[~similar_users_data['book_id'].isin(
                    user_train_data)]
                # print('similar_users_clear')
                # print(similar_users_clear)
                # print(similar_users_clear.shape) #1236

                # Create pivot table for similar_users_clear
                pivot_similar_users_clear = pd.pivot_table(
                    similar_users_clear, values=values_name, index=index_name, columns=column_name, fill_value=0, sort=False)
                # #agar tidak Nan sehingga dapat dilakukan predict rating pada buku yang belum di rating user lain
                # print(pivot_similar_users_clear)

                predicted_rating = pd.DataFrame(
                    columns=['book_id', 'predicted_rating'])
                for i, data in pivot_similar_users_clear.items():
                    # print('i',i)
                    # print('data',data)
                    # print('data',data[i])
                    predict_rating = np.array(pivot_similar_users_clear[i])
                    # print(predict_rating)
                    # print(predict_rating.shape)
                    weight_rating = 0
                    rating = 0
                    sum_similarity = 0
                    counter = 0
                    # print(predict_rating)
                    for j in range(len(predict_rating)):
                        if predict_rating[j] == 0:
                            # print('i',i)
                            # print(predict_rating[j])
                            continue
                            # print(0)
                        else:
                            weight_rating += predict_rating[j] * \
                                similar_users_k.at[j, 'similarity']
                            sum_similarity += similar_users_k.at[j, 'similarity']
                        #     print('rating',rating)
                        #     print('sum_similarity', sum_similarity)
                        #     print('weight_rating',weight_rating)
                    rating = weight_rating/sum_similarity
                    predicted_rating = pd.concat([predicted_rating, pd.DataFrame(
                        {'book_id': [i], 'predicted_rating': [rating]})])
                    # print(predicted_rating)
                # predicted_rating['book_id'] = predicted_rating['book_id']
                predicted_rating = predicted_rating.sort_values(
                    'predicted_rating', ascending=False).reset_index(drop=True)
                # predicted_rating = predicted_rating.merge(books_df[['book_id', 'subject','semester', 'jurusan']], on='book_id', how='left')
                return predicted_rating

                # #Weighted Hybrid
            def weightedHybrid(mahasiswa_id, user_train_data, ratings_data_df, books_data_df,full_data_books, k):
                    usercollab = userCollaborativeFiltering(
                        mahasiswa_id, ratings_data_df, user_train_data, books_data_df, k).head(25)
                    content = contentBased(user_train_data, books_data_df).head(25)
                    normalize1 = usercollab
                    normalize1['predicted_rating'] = normalize1['predicted_rating'].div(
                        5)

                    # print('content',content.dtypes)
                    normalize1 = normalize1.rename(
                        columns={'predicted_rating': 'similarity'})
                    normalize1 = normalize1.sort_values(
                        'similarity', ascending=False)

                    # content['similarity'] =content['similarity']*2
                    # print(content)
                    # weight hybrid
                    hybrid = content.set_index('book_id').add(
                        normalize1.set_index('book_id'), fill_value=0).reset_index()
                    hybrid = hybrid.rename(columns={'similarity': 'final result'})
                    hybrid = hybrid.sort_values(
                        'final result', ascending=False).reset_index(drop=True)
                    hybrid = hybrid.merge(
                        full_data_books[['book_id', 'title',
                        'author', 'publisher', 'subject', 'city', 'year', 'language', 'jurusan']], on='book_id', how='left')
                    # print(mahasiswa_id)
                    # print(user_books_test_all)
                    hybrid['year'] = hybrid['year'].fillna(0).apply(int)
                    # Remove duplicate index 0
                    # hybrid = hybrid.iloc[1:]
                    hybrid = hybrid.drop_duplicates(subset='book_id')
                    print(hybrid)
                    return hybrid
            user_ratings_df_all_book = user_ratings_df_all_book.drop_duplicates(subset='book_id')
        hybrids = weightedHybrid(selected_mahasiswa_id, user_books_train, ratings_df, books_df, full_books_df, 25)

        return render_template('home.html', hybrids=hybrids, mahasiswa=user_ratings_df_all_book, length_books = len(user_ratings_df_all_book), length_rec = len(hybrids) ,head="SISTEM REKOMENDASI BUKU PERPUSTAKAAN", username=username)

@app.route('/booksrating', methods=['POST', 'GET'])
def booksrating():
    conn = connect_db()
    mahasiswa_id_rating = session.get('nomor_induk')
    username = session.get('username')
    books_rated = Collections.query.all()
    if request.method == 'POST':
        nrp_content = mahasiswa_id_rating
        rating_content = json.loads(request.form.getlist('ratedrow')[0])
        print(type(rating_content[0]['knokat']))
        print(rating_content[0]['knokat'])
        rating_table = []
        for rate in rating_content:
            rating_table.append(Ratings(nomor_induk=nrp_content, knokat=rate['knokat'],rating = rate['rating'], kode_buku= rate['kode_buku']))
        print(len(rating_table))
        if len(rating_table) >= 10:
            try:
                db.session.add_all(rating_table)
                db.session.commit()
                flash("Rating buku yang telah kamu berikan sudah direkam, Silahkan kembali menuju HOME!!!")
                return redirect('/booksrating')
            except:
                return "ERROR!!!"
    
    return render_template('booksrating.html', head="SISTEM REKOMENDASI BUKU PERPUSTAKAAN", booksrated=books_rated, confirm_loan=False, username=username)

# @app.route('/collectionloan', methods=['POST', 'GET'])
# def loan():
#     conn = connect_db()
#     username = session.get('username')
#     loans = Collections.query.all()

#     if request.method == 'POST':
#         # retrieve the selected book IDs from the form submission
#         selected_books = request.form.getlist('loanrow')[0]

#         # if no books were selected, display an error message
#         if not selected_books:
#             error_msg = 'Please select at least one book to loan.'
#             return render_template('collectionloan.html', head="SISTEM PEMINJAMAN BUKU PERPUSTAKAAN", loans=loans, error_msg=error_msg)

#         # if the loan is confirmed
#         if 'confirmed' in request.form:

#             # perform the loan operation for the selected books
#             for loan in loans:
#                 if str(loan.kode_buku) in selected_books:
#                     # create a new Circulation record
#                     now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
#                     return_back = datetime.now() + timedelta(days=14)
#                     return_back_formated = return_back.strftime(
#                         '%d/%m/%Y %H:%M:%S')
#                     circ = Circulations(
#                         nomor_induk=session['nomor_induk'],
#                         kode_buku=loan.kode_buku,
#                         tanggal_pesan=now,
#                         tanggal_pinjam=now,
#                         tanggal_batas=return_back_formated,
#                         tanggal_kembali=return_back_formated,
#                         status_pinjam=0,
#                         status_terlambat=0,
#                         bst_bayar='T',
#                         tanggal_bayar=now,
#                         no_kw=2131,
#                         trans_status='delayed payment',
#                         trans_tipe='borrow',
#                         trans_id_ref=3242,
#                         kode_operator='wiros',
#                         tanggal_update=now
#                     )
#                     # add the record to the database
#                     db.session.add(circ)
#                     # commit the changes to the database
#                     db.session.commit()

#             # redirect the user to a success page or back to the loan form
#             return redirect(url_for('history'))

#         # otherwise, display confirmation modal
#         return render_template('collectionloan.html', head="SISTEM PEMINJAMAN BUKU PERPUSTAKAAN", loans=loans, confirm_loan=True)

#     # if request method is GET, display loan form
#     return render_template('collectionloan.html', head="SISTEM PEMINJAMAN BUKU PERPUSTAKAAN", loans=loans, confirm_loan=False, username=username)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('nomor_induk', None)
    if 'nomor_induk' in session and 'username' in session:
        print('Session still has nomor_induk and username')
    else:
        print('Session doesnt have nomor_induk and username')

    return redirect(url_for('login'))


@app.route('/history', methods=['GET', 'POST'])
def history():
    conn = connect_db()
    username = session.get('username')  # ambil username dr session
    circul = Circulations.query.all()
    coll = Collections.query.all()

    # if circulation is not None:
    #     print('1')
    # else:
    #     print('Handle the case where circulation is None')

    joined_data = db.session.query(Collections, Circulations).\
        join(Circulations, Collections.kode_buku == Circulations.kode_buku).\
        all()
    print(joined_data)
    # Create a list of dictionaries to store the joined data
    history_data = []
    for collection, circulation in joined_data:
        # print(collection)
        # print(circulation)
        history_data.append({
            'title': collection.title,
            'author': collection.author,
            'publisher': collection.publisher,
            'kode_buku': collection.kode_buku,
            'knokat': collection.knokat,
            'trans_id': circulation.trans_id,
            'nomor_induk': circulation.nomor_induk,
            'tanggal_pinjam': circulation.tanggal_pinjam,
            'tanggal_pesan': circulation.tanggal_pesan,
            'tanggal_batas': circulation.tanggal_batas,
            'tanggal_kembali': circulation.tanggal_kembali,
            'status_pinjam': circulation.status_pinjam,
            'status_terlambat': circulation.status_terlambat,
            'trans_tipe': circulation.trans_tipe
        })

        if request.method == 'POST':
            rating_content = request.form.getlist('feedbackrow')[0]
            # convert the string to a Python list
            rating_content = json.loads(rating_content)
            print(rating_content)

            for rating in rating_content:
                kode_buku = rating['kode_buku']
                rating_value = rating['rating']
                review_value = rating['review']

                # get the circulation record with the given kode_buku
                circulat = Circulations.query.filter_by(
                    kode_buku=kode_buku).first()

                # check if the circulation record exists and the rating is not set yet
                if circulat and circulat.rating is None:
                    # set the rating
                    circulat.rating = rating_value
                    circulat.review = review_value
                    db.session.commit()
            print('Rating updated successfully')

            # get the list of circulation records for the current user
            history_data = Circulations.query.filter_by(
                nomor_induk=session['nomor_induk']).all()

            for circulation in history_data:
                if circulation.kode_buku == kode_buku and circulation.bst_bayar == 'Y' and circulation.trans_tipe != 'return':
                    extend1 = circulation
                    break
            else:
                extend1 = None

            if extend1:
                print('ya')
                flash('You cant extend this book, Please complete your payment first')
            else:
                loans_count = len([c for c in circul if c.kode_buku == kode_buku and c.nomor_induk ==
                                  session['nomor_induk'] and c.trans_tipe == 'extend'])
                print(loans_count)
                if loans_count == 2:
                    flash('3 Times maximum loan per kode_buku only')
                else:
                    # create a new Circulation record
                    now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                    return_back = datetime.now() + timedelta(days=14)
                    return_back_formated = return_back.strftime(
                        '%d/%m/%Y %H:%M:%S')
                    extends = Circulations(
                        nomor_induk=session['nomor_induk'],
                        kode_buku=kode_buku,
                        tanggal_pesan=now,
                        tanggal_pinjam=now,
                        tanggal_batas=return_back_formated,
                        tanggal_kembali=return_back_formated,
                        status_pinjam=1,
                        status_terlambat=0,
                        bst_bayar='F',
                        tanggal_bayar=now,
                        no_kw=2131,
                        # replace 'trans' with the appropriate attribute name
                        trans_status='delayed payment',
                        trans_tipe='extend',
                        trans_id_ref=3242,
                        kode_operator='wiros',
                        tanggal_update=now
                    )
                    # add the record to the database
                    db.session.add(extends)
                    # commit the changes to the database
                    db.session.commit()
                    return redirect(url_for('history'))

    return render_template('survey.html', survey=coll,  head="SISTEM PEMINJAMAN BUKU PERPUSTAKAAN", username=username)


# @app.route('/survey', methods=['POST', 'GET'])
# def survey():

#     if request.method == 'POST':
#         nrp_content = request.form['nrp']
#         jurusan_content = request.form['jurusan']
#         instagram_content = request.form['instagram']
#         rating_content = json.loads(request.form.getlist('ratedrow')[0])

#         print(type(rating_content[0]['knokat']))
#         print(rating_content[0]['knokat'])
#         for rate in rating_content:
#         rating_table = []
#             rating_table.append(Rating(nomor_induk=nrp_content, jurusan=jurusan_content, instagram = instagram_content, knokat=rate['knokat'],rating = rate['rating'], kode_buku= rate['kode_buku']))
#         print(len(rating_table))
#         if len(rating_table) >= 10:
#             try:
#                 db.session.add_all(rating_table)
#                 db.session.commit()
#                 flash("Rating buku yang telah kamu berikan sudah direkam, Terima kasih atas partisipasinya!!!")
#                 return redirect('/survey')
#             except:
#                 return "ERROR!!!"
#     else:
#         survey = Collections.query.all()
#         return render_template('survey.html', survey=survey, head = "Book Rating Survey")

# def EnableSubmit():
#      sbmt = document.getElementById('submit')
#     if request.form('ratedrow') >= 20:
#         sbmt.disabled = False
#     else:
#          sbmt.disabled = True


# # conn = psycopg2.connect(database="skripsi", user="postgres", password="albert10", host="localhost", port="5432")
# engine = create_engine(
#     'postgresql+psycopg2://postgres:albert10@localhost:5432/skripsi')

# books_df = pd.read_sql_query('SELECT * FROM collections', engine)
# full_books_df = pd.read_sql_query('SELECT * FROM collections', engine)
# full_books_df.rename(columns={'kode_buku': 'book_id'},
#                      inplace=True)  # rename nama kolom
# full_books_df['book_id'] = full_books_df['book_id'].apply(lambda x: x.strip())
# books_df['subject'] = books_df['subject'].fillna(
#     books_df['title'])  # fillna subject dengan title

# # subject preprocessing
# # mengupper casekan semua subject agar perhitungan pada vector tidak salah
# books_df['subject'] = books_df['subject'].str.upper()
# clean_special_char = re.compile('[/(){}\[\]\|@;-]')
# # lambda function digunakan untuk mempassing setiap element dari kolom subject sebagai argument ke sub()
# books_df['subject'] = books_df['subject'].apply(
#     lambda x: clean_special_char.sub(' ', x))
# # replace/subtract subject yg memiliki karakter seperti pada clean_special_char

# # pengganti rbooks
# books_df.drop(['knokat', 'author', 'publisher', 'city', 'year', 'language',
#               'mata_kuliah'], inplace=True, axis=1)  # menghilangkan tabel yang tidak diperlukan
# books_df.rename(columns={'kode_buku': 'book_id'},
#                 inplace=True)  # rename nama kolom
# # menghilangkan data yang duplicates berdasarkan column book_id
# books_df = books_df.drop_duplicates(subset='book_id')
# books_df = books_df.sort_values(by='book_id', ascending=True).reset_index(
#     drop=True)  # mengurutkan kode buku dan mereset indexnya
# books_df['book_id'] = books_df['book_id'].apply(
#     lambda x: x.strip())  # menghilangkan spasi dari data didatabase
# # menghilangkan spasi dari data didatabase
# books_df['title'] = books_df['title'].apply(lambda x: x.strip())
# # print(books_df)

# ratings_df = pd.read_sql_query('SELECT * FROM ratings', engine)
# # ratings_df head()
# # menghilangkan tabel yang tidak diperlukan
# ratings_df.drop('knokat', inplace=True, axis=1)
# ratings_df.rename(columns={'nomor_induk': 'user_id',
#                   'kode_buku': 'book_id'}, inplace=True)
# # rat = ratings_df groupby('nomor_induk').kode_buku.apply(','.join).reset_index()
# ratings_df['user_id'] = ratings_df['user_id'].apply(
#     lambda x: x.strip())  # menghilangkan spasi dari data didatabase
# ratings_df['book_id'] = ratings_df['book_id'].apply(
#     lambda x: x.strip())  # menghilangkan spasi dari data didatabase
# # print(ratings_df)

# selected_user_id = 'C14210208'
# ratings_subject_df = pd.merge(left=ratings_df, right=books_df[[
#                               'book_id', 'subject', 'title']], on='book_id', how='left')

# num_samples = len(ratings_df[ratings_df['user_id']
#                   == selected_user_id]['book_id'].values)
# # print(num_samples)
# test_size = 10/num_samples
# # print(test_size)
# user_rating_train, user_rating_test = train_test_split(list(ratings_df[ratings_df['user_id'] == selected_user_id]['book_id'].values),
#                                                        test_size=test_size, shuffle=False)

# clean_ratings_df = ratings_subject_df.copy()

# for i in range(len(user_rating_test)):
#     clean_ratings_df = clean_ratings_df.drop(clean_ratings_df.loc[(clean_ratings_df.user_id == selected_user_id) &
#                                                                   (clean_ratings_df.book_id == user_rating_test[i])]
#                                              .index)
# # content-based filtering


# def content_based(all_rated_train, rated_books_df):
#     # merge all_rated_train(user_rated_train) book_id dengan subjectnya
#     merge_rated_book_subject = pd.merge(
#         pd.Series(all_rated_train, name='book_id'), rated_books_df, on='book_id')
#     print(all_rated_train)
#     print(merge_rated_book_subject)

#     # #merge all_rated_book_subject
#     # menyatukann semua subject dari all_rated_train selected_user
#     all_rated_book_subject = ','.join(merge_rated_book_subject.subject)
#     # print(all_rated_book_subject)
#     all_rated_book_subject_df = pd.DataFrame(
#         {'subject': all_rated_book_subject}, index=[0])  # dimasukkan ke dalam dataframe
#     # print(all_rated_book_subject_df)

#     # tf-idf
#     tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
#     # (khusus untuk data training)untuk digunakan pada data pelatihan sehingga kita dapat menskalakan data pelatihan dan juga mempelajari parameter penskalaan dari data tersebut.
#     user_rated_train_transform = tfidf.fit_transform(
#         all_rated_book_subject_df['subject'])
#     # print(user_rated_train_transform)
#     # Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data.
#     all_book = tfidf.transform(rated_books_df['subject'])
#     # print(all_book)

#     # cosine similarity
#     sim = cosine_similarity(user_rated_train_transform, all_book)
#     # print(sim)
#     sim = sim.transpose()  # transpose matrix agar berbentuk vertikal column sehingga dapat
#     # print(sim)
#     similarity_df = pd.DataFrame(columns=['book_id', 'similarity'])
#     for i in range(len(rated_books_df)):
#         similarity_df = pd.concat([similarity_df, pd.DataFrame({'book_id': [
#                                   rated_books_df['book_id'][i]], 'similarity': [sim[i]]})], ignore_index=True)
#     # print(similarity_df)
#     for i in range(len(all_rated_train)):
#         # drop book_id dari all_rated_train
#         similarity_df = similarity_df.drop(
#             index=similarity_df[similarity_df['book_id'] == all_rated_train[i]].index, errors='ignore')
#     similarity_df = similarity_df.sort_values('similarity', ascending=False).reset_index(
#         drop=True)  # sort data dari nilai similarity terbesar
#     return similarity_df

# # USER COLLABORATIVE FILTERING FUNCTION FIX


# def userCollaborativeFiltering(user_id, user_rating_data, ratings_data_df, k):

#     column_name = 'book_id'
#     index_name = 'user_id'
#     values_name = 'rating'

#     # userid=ratings_data_df.loc[:,index_name] #cara mengakses row dengan loc
#     # print(userid)

#     # Create pivot table for book rating
#     pivot_rating = pd.pivot_table(
#         ratings_data_df, values=values_name, index=index_name, columns=column_name)
#     # untuk mengambil rata" kolom yang mana adalah seharusnya rata" rating user maka pivot_rating ditranpose agar user_id menjadi column
#     pivot_rating = pivot_rating.T
#     # fill na with pivot_rating mean per user, karena agar tidak bias
#     pivot_rating = pivot_rating.fillna(pivot_rating.mean())
#     # print(pivot_rating)
#     # pivot_rating dikembalikan kebentuk asalnya untuk proses rekomendasi
#     pivot_rating = pivot_rating.T
#     # print(pivot_rating)
#     # print(pivot_rating.loc[userid[1]])
#     # COSINE SIMILARITY
#     # Calculate similarity between selected user and others

#     # sim=cosine_similarity([pivot_rating.loc[user_id]], [pivot_rating.loc[userid[3000]]]) # cosine_similarity dengan loc kiri selected_user, kanan other_user
#     # print([pivot_rating.loc[userid[3000]]])
#     # print(sim)

#     # Get user's book rating
#     user_rating = pivot_rating.loc[user_id]
#     # print('1')
#     # print(user_rating)

#     # Drop user_rating from pivot_rating
#     others_rating = pivot_rating.copy()
#     others_rating = others_rating.drop(index=[user_id])
#     # print(others_rating)
#     # others_rating = others_rating.loc['C14180025']
#     # print(others_rating)

#     # Create dataframe to contain similarity results
#     similarity_result_df = pd.DataFrame(columns=['userId', 'similarity'])

#     # # Iterate through others_rating
#     for other_user, mean in others_rating.iterrows():
#         # print(other_user)
#         # other_user_rating = others_rating.loc[userid[201]]
#         # print(other_user_rating)
#         sim = cosine_similarity([user_rating], [others_rating.loc[other_user]])
#         # print(sim)
#         # similarity_result_df = similarity_result_df.append({'bookId': other_user, 'similarity': cosinesim}, ignore_index=True)
#         similarity_result_df = pd.concat([similarity_result_df, pd.DataFrame(
#             {'userId': [other_user], 'similarity': [sim]})], ignore_index=True)
#     # print(similarity_result_df)

#     # # Sort data by similarity and get k data
#     similar_users = similarity_result_df.sort_values(
#         'similarity', ascending=False).reset_index(drop=True).loc[similarity_result_df.index < k]
#     # print(similar_users)

#     # # Get book_id and rating by user_id from similar_users
#     similar_users_data = pd.DataFrame(
#         columns=[index_name, column_name, values_name])
#     for i in range(len(similar_users.index)):
#         # similar_users_data = similar_users_data.append(ratings_data_df.loc[ratings_data_df['book_id']== similar_users.at[i, 'bookId']])
#         similar_users_data = pd.concat([similar_users_data, pd.DataFrame(
#             ratings_data_df.loc[ratings_data_df['user_id'] == similar_users.at[i, 'userId']])])
#     # print(similar_users_data)

#     # # Drop books rated by selected user
#     for i in range(len(user_rating_data)):
#         similar_users_data = similar_users_data.drop(index=similar_users_data[similar_users_data['book_id'] == user_rating_data[i]].index,
#                                                      errors='ignore')
#     # print(similar_users_data)

#     # # Create pivot table for similar_users_data
#     pivot_similar_users_data = pd.pivot_table(
#         similar_users_data, values=values_name, index=index_name, columns=column_name, fill_value=0)
#     # print(pivot_similar_users_data)

#     # # Create dataframe to contain predicted ratings #cek predict rating
#     predicted_rating = pd.DataFrame(columns=['book_id', 'predicted_rating'])
# #   predicted = pd.DataFrame(columns=['book_id', 'predicted_rating'])
#     for i, data in pivot_similar_users_data.items():
#         # print(i)
#         predict_rating = np.array(pivot_similar_users_data[i])
#         # print(predict_rating)
#         rating = 0
#         counter = 0
#         for j in predict_rating:
#             # print(j)
#             if predict_rating[j] == 0:
#                 continue
#             rating += predict_rating[j]*similar_users.at[j, 'similarity']
#             counter += 1
#         if counter != 0:
#             rating = rating/counter
#         #   predicted_rating = pd.concat([predicted_, pd.DataFrame({'book_id': [i], 'predicted_rating': [rating]})], ignore_index=True)
#         #   merged_df = predicted_rating.merge(ratings_subject_df[['book_id', 'title']], on='book_id', how='left')
#         # print(merged_df)
#         # filter for the current book_id and get the subject value
#         #   subject = merged_df[merged_df['book_id'] == i]['title'].values[0]
#         predicted_rating = pd.concat([predicted_rating, pd.DataFrame(
#             {'book_id': [i], 'predicted_rating': [rating]})], ignore_index=True)
#     predicted_rating['book_id'] = predicted_rating['book_id']
#     predicted_rating = predicted_rating.sort_values(
#         'predicted_rating', ascending=False).reset_index(drop=True)
#     print(predicted_rating)
#     return predicted_rating


# def itemCollaborativeFiltering(user_id, user_rating_data, ratings_data_df, k):
#     column_name = 'book_id'
#     index_name = 'user_id'
#     values_name = 'rating'

#     # Get the books that the selected user has not rated
#     # mengambil book_id yang tidak ada dimiliki user_rating_data yang terdapat pada data set rating_df (A-B)
#     unrated_books = set(ratings_data_df['book_id']).difference(
#         set(user_rating_data))
#     # print(len(unrated_books))

#     # Create a pivot table of user ratings data
#     pivot_ratings_data = ratings_data_df.pivot(
#         values=values_name, index=index_name, columns=column_name)
#     pivot_ratings_data = pivot_ratings_data.fillna(pivot_ratings_data.mean())
#     pivot_ratings_data = pivot_ratings_data.T
#     # print(pivot_ratings_data)
#     # Compute item-item similarity matrix using cosine similarity
#     # mencari similarity antar buku (Semua buku)
#     item_similarity_matrix = cosine_similarity(pivot_ratings_data)
#     # print(item_similarity_matrix)
#     # Create a dictionary to store the similarity scores for each unrated book
#     predicted_ratings = {}

#     # Compute the predicted ratings for each unrated book
#     # agar lebih mudah diakses dari set (dikeluarkan dari set sehingga dapat lebih mudah digunakan)
#     for book_id in unrated_books:
#         # Compute the weighted average rating for the current book based on the user's ratings
#         weighted_sum = 0
#         similarity_sum = 0
#         # print(book_id)
#         # print('--------')
#         # print(pivot_ratings_data.index.get_loc(book_id))
#         for rated_book_id in user_rating_data:
#             # Check if the similarity score between the current book and the rated book is greater than 0
#             # print(pivot_ratings_data.index.get_loc(rated_book_id))

#             if item_similarity_matrix[pivot_ratings_data.index.get_loc(book_id), pivot_ratings_data.index.get_loc(rated_book_id)] > 0:
#                 # untuk cek dan mencari dari buku yang belum dirating user similarity antara dua buku, antara buku yg blm dirating dan yg telah dirating user(user_training_data), lebih besar dari nol.
#                 # digunakan index agar dapat mengambil elemen(value) similarity dari item_similarity_matrix secara tepat
#                 # mengapa similarity dari dua buku, karena similaritynya sudah ditemukan nilainya sudah ternormalisasi sehingga lebih mudah atau cepat untuk membandingkannya

#                 weighted_sum += item_similarity_matrix[pivot_ratings_data.index.get_loc(book_id), pivot_ratings_data.index.get_loc(
#                     rated_book_id)] * ratings_data_df[(ratings_data_df['user_id'] == user_id) & (ratings_data_df['book_id'] == rated_book_id)]['rating'].values[0]
#                 # print(weighted_sum)
#                 similarity_sum += item_similarity_matrix[pivot_ratings_data.index.get_loc(
#                     book_id), pivot_ratings_data.index.get_loc(rated_book_id)]
#                 # print(similarity_sum)
#         if similarity_sum > 0:
#             predicted_ratings[book_id] = weighted_sum / similarity_sum
#             # print(predicted_ratings[book_id])
#     # # Create a DataFrame of the predicted ratings
#     predicted_ratings_df = pd.DataFrame(list(predicted_ratings.items()), columns=[
#                                         'book_id', 'predicted_rating'])

#     # Sort the predicted ratings by descending order of rating value
#     predicted_ratings_df.sort_values(
#         'predicted_rating', ascending=False, inplace=True)
#     predicted_ratings_df.reset_index(inplace=True, drop=True)
#     # Return the top k recommendations
#     # print(selected_user_id)
#     return predicted_ratings_df.head(k)

# # #Weighted Hybrid


# def weightedHybrid(user_id, user_rating_data, ratings_data_df, rated_books_df, full_columns_books, k):
#     usercollab = userCollaborativeFiltering(
#         user_id, user_rating_data, ratings_data_df, k)
#     itemcollab = itemCollaborativeFiltering(
#         user_id, user_rating_data, ratings_data_df, k)
#     content = content_based(user_rating_data, rated_books_df)
#     normalize1 = usercollab
#     normalize2 = itemcollab
#     normalize1['predicted_rating'] = normalize1['predicted_rating'].div(5)
#     normalize2['predicted_rating'] = normalize2['predicted_rating'].div(5)
#     # print(normalize['predicted_rating'])
#     normalize1 = normalize1.rename(columns={'predicted_rating': 'similarity'})
#     normalize2 = normalize2.rename(columns={'predicted_rating': 'similarity'})
#     normalize1 = normalize1.sort_values(
#         'similarity', ascending=False).reset_index(drop=True)
#     normalize2 = normalize2.sort_values(
#         'similarity', ascending=False).reset_index(drop=True)
#     # print(normalize1)
#     hybrid = content.set_index('book_id').add(
#         normalize1.set_index('book_id'), fill_value=0).reset_index()
#     hybrid = hybrid.set_index('book_id').add(
#         normalize2.set_index('book_id'), fill_value=0).reset_index()
#     hybrid = hybrid.rename(columns={'similarity': 'final_result'})
#     hybrid = hybrid.sort_values(
#         'final_result', ascending=False).reset_index(drop=True)
#     top_recommendation = hybrid.head(10)
#     # join/merged top_recommendation with full books_df agar dapat menambahkan info" buku lainya
#     top_recommendation = top_recommendation.merge(
#         full_columns_books, on='book_id', how='left')
#     print(top_recommendation[['book_id', 'final_result', 'title',
#           'author', 'publisher', 'subject', 'city', 'year', 'language']])
#     return top_recommendation[['book_id', 'final_result', 'title', 'author', 'publisher', 'subject', 'city', 'year', 'language']]


# # userCollaborativeFiltering(selected_user_id, user_rating_train, clean_ratings_df, 100)
# weightedHybrid(selected_user_id, user_rating_train,
#                clean_ratings_df, books_df, full_books_df, 100)

if __name__ == "__main__":
    app.run(debug=True)

# def index():
#     if request.method == 'POST':
#         task_content = request.form['content']
#         new_task = Todo(content=task_content)

#         try:
#             db.session.add(new_task)
#             db.session.commit()
#             return redirect('/')

#         except:
#             return "There was an issue adding your task"

#     else:
#         tasks = Todo.query.order_by(Todo.author).all()
#         return render_template('index.html', tasks=tasks)

# @app.route('/delete/<int:id>')
# def delete(id):
#     task_to_delete = Todo.query.get_or_404(id)

#     try:
#         db.session.delete(task_to_delete)
#         db.session.commit()
#         return redirect('/')
#     except:
#         return 'There was a problem deleting that task'

# @app.route('/update/<int:id>', methods=['GET', 'POST'])
# def update(id):
    # task_update = Todo.query.get_or_404(id)
    # if request.method == 'POST':
    #     task_update.content = request.form['content']
    #     try:
    #         db.session.commit()
    #         return redirect('/')
    #     except:
    #         return 'There was an issu updating your task'
    # else:
    #     return render_template('update.html', task=task_update)
