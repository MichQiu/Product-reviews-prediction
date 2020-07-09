import pandas as pd
import numpy as np
import unidecode as ud
import re
import stanza
from datetime import datetime
from copy import deepcopy
from math import sqrt, isnan
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.naive_bayes import BernoulliNB as BNB

# read the files
file = ['/olist_customers_dataset', '/olist_geolocation_dataset', '/olist_order_items_dataset',
        '/olist_order_payments_dataset', '/olist_order_reviews_dataset', '/olist_orders_dataset',
        '/olist_products_dataset', '/olist_sellers_dataset', '/product_category_name_translation']

df_customers = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[0] + ".csv")
df_geolocation = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[1] + '.csv')
df_order_items = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[2] + '.csv')
df_order_payments = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[3] + '.csv')
df_order_reviews = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[4] + '.csv')
df_orders = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[5] + '.csv')
df_products = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[6] + '.csv')
df_sellers = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[7] + '.csv')
df_translation = pd.read_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce' + file[8] + '.csv')


# converting dates
def convert_time(data, feature, order='dmy', format='minutes'):
    for i in range(0, len(data[feature])):
        if type(data.loc[i, feature]) == float:
            pass
        else:
            if order == 'dmy':
                if format == 'minutes':
                    datetime_object = datetime.strptime(data.loc[i, feature], '%d/%m/%Y %H:%M')
                    data.loc[i, feature] = datetime_object
                elif format == 'seconds':
                    datetime_object = datetime.strptime(data.loc[i, feature], '%d/%m/%Y %H:%M:%S')
                    data.loc[i, feature] = datetime_object
            elif order == 'ymd':
                if format == 'minutes':
                    datetime_object = datetime.strptime(data.loc[i, feature], '%Y-%m-%d %H:%M')
                    data.loc[i, feature] = datetime_object
                elif format == 'seconds':
                    datetime_object = datetime.strptime(data.loc[i, feature], '%Y-%m-%d %H:%M:%S')
                    data.loc[i, feature] = datetime_object

# time difference calculation function
def time_duration(data, feature_1, feature_2, new_feature):
    for i in range(0, len(data[feature_1])):
        if pd.isna(data.loc[i, feature_1]) or pd.isna(data.loc[i, feature_2]) == True:
            pass
        elif data.loc[i, feature_1] > data.loc[i, feature_2]:
            timedelta = data.loc[i, feature_1] - data.loc[i, feature_2]
            data.loc[i, new_feature] = timedelta.days
        else:
            timedelta = data.loc[i, feature_2] - data.loc[i, feature_1]
            data.loc[i, new_feature] = timedelta.days

# converting dates and calculating time difference
convert_time(df_order_reviews, 'review_creation_date', format='minutes')
convert_time(df_order_reviews, 'review_answer_timestamp', format='minutes')
df_order_reviews['review_duration'] = np.nan
time_duration(df_order_reviews, 'review_creation_date', 'review_answer_timestamp', 'review_duration')
df_order_reviews = df_order_reviews.drop(['review_creation_date', 'review_answer_timestamp'], axis=1)

convert_time(df_orders, 'order_estimated_delivery_date', order='ymd', format='seconds')
convert_time(df_orders, 'order_delivered_customer_date', order='ymd', format='seconds')
convert_time(df_orders, 'order_purchase_timestamp', order='ymd', format='seconds')
df_orders['PurchaseToDelivered_duration'] = np.nan
df_orders['delivery_expectation_diff'] = np.nan
time_duration(df_orders, 'order_delivered_customer_date', 'order_purchase_timestamp', 'PurchaseToDelivered_duration')
time_duration(df_orders, 'order_estimated_delivery_date', 'order_delivered_customer_date', 'delivery_expectation_diff')
df_orders['PurchaseToDelivered_duration'] = df_orders['PurchaseToDelivered_duration'].fillna(df_orders['PurchaseToDelivered_duration'].mean())
df_orders['delivery_expectation_diff'] = df_orders['delivery_expectation_diff'].fillna(df_orders['delivery_expectation_diff'].mean())
df_orders = df_orders.drop(['order_estimated_delivery_date', 'order_delivered_customer_date', 'order_purchase_timestamp'], axis=1)

# dates
df_review = df_order_reviews[['order_id', 'review_comment_message', 'review_duration']]
df_review = df_review.groupby(df_review['order_id']).sum()
df_estimate_delivery = df_orders[['order_id', 'PurchaseToDelivered_duration', 'delivery_expectation_diff']]
df_estimate_delivery = df_estimate_delivery.groupby(df_estimate_delivery['order_id']).sum()


# remove non-ASCII characters
regexp = re.compile(r'[^\x00-\x7f]')
for i in range(0, len(df_geolocation['geolocation_city'])):
    city = df_geolocation.loc[i, 'geolocation_city']
    if regexp.search(city):
        df_geolocation.loc[i, 'geolocation_city'] = ud.unidecode(city)
    else:
        pass


# sentiment analysis on review comments
stanza.download('pt', processors='tokenize, mwt, pos', package='GSD')
nlp = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos', package='GSD', use_gpu=True, verbose=True)

for i in range(0, len(df_order_reviews['review_comment_message'])):
    comments = df_order_reviews.loc[i, 'review_comment_message']
    if pd.isna(comments):
        pass
    else:
        doc = nlp(comments)
        for sent in doc.sentences:
            word_length = len(sent.words)
            for word in sent.words:
                if word.feats == None:
                    pass
                else:
                    feat_list = word.feats.split('|')
                    count = 0
                    for feat in feat_list:
                        if 'Polarity' in feat:
                            polar = feat[9:]
                            if polar == 'Pos':
                                count += 1
                            elif polar == 'Neg':
                                count -= 1
                        else:
                            pass
                    df_order_reviews.loc[i, 'review_comment_message'] = count / word_length


# transfer df for model 1 to a df for model 1
def transfer_to_review(data):
    df_order_id_transfer= deepcopy(df_order_reviews['order_id'])
    dic_order_id = pd.Series(df_order_id_transfer.order_id.values, index=df_order_id_transfer.order_id.values).to_dict()
    new_df = data['order_id'].map(dic_order_id)
    new_df = new_df.dropna()
    return new_df

# target for model 2
df_review_score = df_order_reviews[['order_id', 'review_score', 'review_duration']]
df_review_score = df_review_score.sort_values(by="order_id")

for i in df_review_score['review_score']:
    if i < 4:
        df_review_score.iloc[i, 1] = 0
    elif i >= 4:
        df_review_score.iloc[i, 1] = 1

# comment or not for model 2
df_order_reviews['review_comment_message'] = np.where(df_order_reviews['review_comment_message'].str.len() > 2, 1, 0)
df_comment = df_order_reviews[['order_id', 'review_comment_message']]
df_comment = df_comment.sort_values(by="order_id")

# set up dictionaries
dic_product_id_to_category = pd.Series(df_products.product_category_name.values, index=df_products.product_id).to_dict()
dic_product_id_to_dis_len = pd.Series(df_products.product_description_lenght.values,
                                      index=df_products.product_id).to_dict()
dic_product_id_to_photo = pd.Series(df_products.product_photos_qty.values, index=df_products.product_id).to_dict()
dic_seller_id_to_city = pd.Series(df_sellers.seller_city.values, index=df_sellers.seller_id).to_dict()
dic_order_id_to_customer_id = pd.Series(df_orders.customer_id.values, index=df_orders.order_id).to_dict()
dic_customer_id_to_city = pd.Series(df_customers.customer_city.values, index=df_customers.customer_id).to_dict()
dic_name_translation = pd.Series(df_translation.product_category_name_english.values,
                                 index=df_translation.product_category_name).to_dict()

# distance from seller to costumer for both
df_geolocation = df_geolocation.drop(['geolocation_zip_code_prefix', 'geolocation_state'], axis=1)
df_geolocation.groupby("geolocation_city").mean()
dic_city_to_lat = pd.Series(df_geolocation.geolocation_lat.values, index=df_geolocation.geolocation_city).to_dict()
dic_city_to_lng = pd.Series(df_geolocation.geolocation_lng.values, index=df_geolocation.geolocation_city).to_dict()
df_distance = df_order_items.drop(['order_item_id', 'shipping_limit_date', 'price', 'freight_value', 'product_id'],
                                  axis=1)
df_order_id = deepcopy(df_distance['order_id'])

df_distance['order_id'].replace(dic_order_id_to_customer_id, inplace=True)
df_distance['order_id'].replace(dic_customer_id_to_city, inplace=True)
df_distance.rename(columns={'order_id': 'customer_city'}, inplace=True)
df_distance['seller_id'].replace(dic_seller_id_to_city, inplace=True)
df_distance.rename(columns={'seller_id': 'seller_city'}, inplace=True)
df_distance = pd.concat([df_order_id, df_distance], axis=1)
df_distance["distance"] = np.nan
for a in range(len(df_distance)):
    if (df_distance.iloc[a, 1] not in dic_city_to_lat.keys()) or (df_distance.iloc[a, 2] not in dic_city_to_lat.keys()):
        pass
    else:
        df_distance.iloc[a, 3] = sqrt(pow((dic_city_to_lat[df_distance.iloc[a, 1]]
                                           - dic_city_to_lat[df_distance.iloc[a, 2]]), 2)
                                      + pow((dic_city_to_lng[df_distance.iloc[a, 1]]
                                             - dic_city_to_lng[df_distance.iloc[a, 2]]), 2))
df_distance['distance'] = df_distance['distance'].fillna(df_distance['distance'].mean())
df_distance = df_distance.drop(['customer_city', 'seller_city'], axis=1)
df_distance = df_distance.groupby("order_id").mean()
df_distance = df_distance.sort_values(by="order_id")

# translate names
df_products['product_category_name'].replace(dic_name_translation, inplace=True)

# category name for both
df_cate = df_order_items.drop(['order_item_id', 'shipping_limit_date', 'price', 'freight_value', 'seller_id'], axis=1)
df_cate['product_id'].replace(dic_product_id_to_category, inplace=True)
df_cate.rename(columns={'product_id': 'product_category'}, inplace=True)
df_cate_dum = pd.get_dummies(df_cate['product_category'])
df_cate = pd.concat([df_cate, df_cate_dum], axis=1)
df_cate = df_cate.drop(['product_category'], axis=1)
df_cate = df_cate.groupby("order_id").sum()
df_cate = df_cate.sort_values(by="order_id")

# description length for both
df_description_length = df_order_items.drop(
    ['order_item_id', 'shipping_limit_date', 'price', 'freight_value', 'seller_id'], axis=1)
df_description_length['product_id'].replace(dic_product_id_to_dis_len, inplace=True)
df_description_length.rename(columns={'product_id': 'discription_length'}, inplace=True)
df_description_length = df_description_length.groupby("order_id").mean()
df_description_length = df_description_length.sort_values(by="order_id")

# photos
df_photo = df_order_items.drop(['order_item_id', 'shipping_limit_date', 'price', 'freight_value', 'seller_id'], axis=1)
df_photo['product_id'].replace(dic_product_id_to_photo, inplace=True)
df_photo.rename(columns={'product_id': 'photo'}, inplace=True)
df_photo = df_photo.groupby("order_id").mean()
df_photo = df_photo.sort_values(by="order_id")

# order status
df_orderstatus = df_orders[['order_id', 'order_status']]
encoder = OneHotEncoder(sparse=False)
encoder_fit = encoder.fit_transform(df_orderstatus['order_status'].values.reshape(-1, 1))
encoder_fit = encoder_fit.sort_values(by="order_id")

# freight/payment value
df_fv = df_order_items[['order_id', 'freight_value']]
df_pv = df_order_payments[['order_id', 'payment_value']]

grouped = df_fv['freight_value'].groupby(df_fv['order_id']).sum()
grouped2 = df_pv['payment_value'].groupby(df_pv['order_id']).sum()
set1 = set(grouped.index)
set2 = set(grouped2.index)
new_set = set1.intersection(set2)
new_list = list(new_set)
percentage_fp = {x: grouped[x] / grouped2[x] for x in new_list}
new_percentage_fp = pd.DataFrame({'percentage': list(percentage_fp.values())}, index=percentage_fp.keys())

# find the missing order_id due to the intersection sets
df_a = df_photo.reset_index()
df_b = new_percentage_fp.reset_index()
for order in df_a['order_id']:
    if order not in df_b['order_id']:
        missing_order = order
        break

# concatenate the missing order to freight/payment data
missing_order_id = pd.Series(data=np.nan, index=['bfbd0f9bdef84302105ad712db648a6c'])
new_percentage_fp = pd.concat([new_percentage_fp, missing_order_id])
new_percentage_fp.drop(0, axis=1)
new_percentage_fp['percentage'].fillna(new_percentage_fp['percentage'].mean())
new_percentage_fp = new_percentage_fp.reset_index()
new_percentage_fp = new_percentage_fp.sort_values(by="index")
new_percentage_fp.set_index(new_percentage_fp['index'], inplace=True)
new_percentage_fp = new_percentage_fp.drop(['index', 0], axis=1)

# target for model 1
df_review_or_not = df_orders['order_id']
list1 = []
list2 = df_review_score["order_id"].tolist()

for a in range(len(df_review_or_not)):
    if df_review_or_not[a] in list2:
        list1.append(1)
    else:
        list1.append(0)

df_review_boolean = pd.DataFrame({'review_or_not': list1})
df_review_or_not = pd.concat([df_review_or_not, df_review_boolean], axis=1)
df_review_or_not = df_review_or_not.sort_values(by="order_id")
df_order_id_target = deepcopy(df_photo)
df_order_id_target = df_order_id_target.reset_index()
dic_order_id_target = pd.Series(df_order_id_target.order_id.values, index=df_order_id_target.order_id.values).to_dict()
df_review_or_not['order_id'] = df_review_or_not['order_id'].map(dic_order_id_target)
df_review_or_not = df_review_or_not.dropna()
df_review_or_not.set_index(df_review_or_not['order_id'], inplace=True)
df_review_or_not = df_review_or_not.drop(['order_id'], axis=1)

# merge datasets for model 1
df_distance

df_orderstatus['order_id'] = df_orderstatus['order_id'].map(dic_order_id_target)
df_orderstatus = df_orderstatus.dropna()
df_orderstatus.set_index(df_orderstatus['order_id'], inplace=True)
df_orderstatus = df_orderstatus.drop(['order_id'], axis=1)
df_orderstatus = pd.get_dummies(df_orderstatus['order_status'])
df_orderstatus = df_orderstatus.sort_values(by="order_id")

grouped2 = grouped2.reset_index()
grouped2['order_id'] = grouped2['order_id'].map(dic_order_id_target)
grouped2 = grouped2.dropna()
grouped2.set_index(grouped2['order_id'], inplace=True)
grouped2 = grouped2.drop(['order_id'], axis=1)
missing_order_id = pd.Series(data=np.nan, index=['bfbd0f9bdef84302105ad712db648a6c'])
grouped2 = pd.concat([grouped2, missing_order_id])
grouped2.drop(0, axis=1)
grouped2['payment_value'].fillna(grouped2['payment_value'].mean())
grouped2 = grouped2.reset_index()
grouped2 = grouped2.sort_values(by="index")
grouped2.set_index(grouped2['index'], inplace=True)
grouped2 = grouped2.drop(['index', 0], axis=1)


# model 2 mappings
df_review_num = deepcopy(df_review_score)
dic_order_id_target2 = pd.Series(df_review_num.order_id.values, index=df_review_num.order_id.values).to_dict()

df_distance2 = deepcopy(df_distance)
df_freight_price = deepcopy(grouped)
df_estimate_delivery2 = deepcopy(df_estimate_delivery)
df_cate2 = deepcopy(df_cate)
df_description_length2 = deepcopy(df_description_length)
df_photo2 = deepcopy(df_photo)

def mapping(data, col):
    x = data.reset_index()
    x[col] = df_review_num[col].map(dic_order_id_target2)
    x = x.dropna()
    x.set_index(x[col], inplace=True)
    x = x.drop([col], axis=1)
    return x

df_distance2 = mapping(df_distance2, 'order_id')
df_freight_price = mapping(df_freight_price, 'order_id')
df_estimate_delivery2 = mapping(df_estimate_delivery2, 'order_id')
df_cate2 = mapping(df_cate2, 'order_id')
df_description_length2 = mapping(df_description_length2, 'order_id')
df_photo2 = mapping(df_photo2, 'order_id')

missing_order2 = []
for order in np.array(df_review_num['order_id']):
    if order not in np.array(df_description_length2['order_id']):
        missing_order2.append(order)
    else:
        pass

for i in missing_order2:
    missing_order_id = pd.Series(data=np.nan, index=[i])
    df_description_length2 = pd.concat([df_description_length2, missing_order_id])
    df_description_length2.drop(0, axis=1)
    df_description_length2['discription_length'].fillna(df_description_length2['discription_length'].mean())
    df_description_length2 = df_description_length2.reset_index()
    df_description_length2 = df_description_length2.sort_values(by="index")
    df_description_length2.set_index(df_description_length2['index'], inplace=True)
    df_description_length2 = df_description_length2.drop(['index', 0], axis=1)


# merge
df_model1 = pd.concat([df_review_or_not, df_distance, grouped, grouped2, df_orderstatus, df_estimate_delivery,
                       df_description_length, df_photo], axis=1)
df_model1['discription_length'] = df_model1['discription_length'].fillna(df_model1['discription_length'].mean())
df_model1['photo'] = df_model1['photo'].fillna(df_model1['photo'].mean())
df_model1 = df_model1.dropna()
df_model1.to_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce/model1.csv')

df_review_score.set_index(df_review_score['order_id'], inplace=True)
df_review_score = df_review_score.drop(['order_id'], axis=1)
df_comment.set_index(df_comment['order_id'], inplace=True)
df_comment = df_comment.drop(['order_id'], axis=1)

df_review_score = df_review_score.join([df_comment, df_distance2, df_freight_price, df_estimate_delivery2], how='right')
x = pd.concat([df_comment, df_distance2, df_freight_price, df_estimate_delivery2], axis=1)
df_model2 = deepcopy(df_review_score)
df_model2.drop_duplicates(inplace=True)
df_model2 = df_model2.dropna()

df_model2.to_csv('/home/mich_qiu/PycharmProjects/DSML/Data_project_2/brazilian-ecommerce/model2.csv')

df_model2 = pd.read_csv('/home/mich_qiu/Downloads/model2.csv')
df_model2.drop(['order_id'], axis=1, inplace=True)

df_model1_target = df_model1['review_or_not']
df_model1_base = df_model1.drop(['review_or_not'], axis=1)

df_model2_target = df_model2['review_score']
df_model2_base = df_model2.drop(['review_score'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_model2_base, df_model2_target, test_size = 0.2, random_state=5)


def crossValid(model, hyp, scores):
    for score in scores:
        print("# Tuning hyperparameters for %s" % score)
        print("\n")
        clf = GridSearchCV(model, hyp, cv=5,
                           scoring=score)
        clf.fit(X_train, Y_train)
        print("Best parameters set found on the training set:")
        print(clf.best_params_)
        print("\n")

lr = LR(C=10, max_iter=200, penalty='l1', solver='liblinear')
lr_fit = lr.fit(X_train, Y_train)
train_score = lr.score(X_train, Y_train)
print(train_score)

# prediction
predicted = lr.predict(X_test)
test_score = lr.score(X_test, Y_test)
print(test_score)

# cross-validation
tuned_parameters = [{'penalty': ['l2'], 'solver': ['saga'],
                     'C': [1, 10, 100],  'max_iter': [200, 300]},
                    {'penalty': ['l1'],
                     'C': [0.5, 1, 1.5, 10, 100], 'solver': ['liblinear'], 'max_iter': [200, 300]}]

scores = ['accuracy', 'f1_macro']
crossValid(lr, tuned_parameters, scores)

dtc = DTC(criterion = 'entropy', max_depth = 7, max_features = None, min_samples_split = 3)
dtc_fit = dtc.fit(X_train, Y_train)
train_score = dtc.score(X_train, Y_train)
print(train_score)

# prediction
predicted = lr.predict(X_test)
test_score = lr.score(X_test, Y_test)
print(test_score)

bnb = BNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior='bool')
bnb_fit = bnb.fit(X_train, Y_train)
train_score = bnb.score(X_train, Y_train)
print(train_score)

# prediction
predicted = bnb.predict(X_test)
test_score = bnb.score(X_test, Y_test)
print(test_score)

xgb_clf = xgb.XGBClassifier(n_jobs=8, objective='binary:logistic', colsample_bytree=0.5, learning_rate=0.5, max_depth=3,
                            n_estimators=25)

xgb_clf.fit(X_train, Y_train)
train_score = xgb_clf.score(X_train, Y_train)
print(train_score)

# prediction
predicted = xgb_clf.predict(X_test)
test_score = xgb_clf.score(X_test, Y_test)
print(test_score)



# cross-validation
tuned_parameters = [{'n_estimators':[25, 50, 75], 'learning_rate':[0.25, 0.5, 0.75], 'colsample_bytree':[0.1, 0.3, 0.5]
                     , 'objective':['binary:logistic'], 'max_depth':[3, 5, 7]}]

scores = ['accuracy', 'f1_macro']
crossValid(xgb_clf, tuned_parameters, scores)


# reparameterise models with new hyperparameters
clf1 = LR(max_iter=200, C=1)
clf2 = DTC(criterion = 'entropy', max_depth = 7, max_features = None, min_samples_split = 3)
clf3 = xgb.XGBClassifier(n_jobs=8, objective='binary:logistic', colsample_bytree=0.1, learning_rate=0.25, max_depth=3,
                            n_estimators=2)

# fit model
eclf1 = VotingClassifier([('lr', clf1), ('dtc', clf2), ('xgb', clf3)], voting='soft')
eclf1 = eclf1.fit(X_train, Y_train)
train_score = eclf1.score(X_train, Y_train)
print(train_score)

# prediction
predicted = eclf1.predict(X_test)
test_score = eclf1.score(X_test, Y_test)
print(test_score)


clf1 = LR(max_iter=200)
clf2 = DTC(criterion = 'gini', max_depth = 3, max_features = 'sqrt', min_samples_split = 7)
clf3 = xgb.XGBClassifier(n_jobs=8, objective='binary:logistic', colsample_bytree=0.5, learning_rate=0.5, max_depth=3,
                            n_estimators=25)
clf4 = BNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior='bool')

# fit model
eclf1 = VotingClassifier([('lr', clf1), ('dtc', clf2), ('xgb', clf3), ('bnb', clf4)], voting='soft')
eclf1 = eclf1.fit(X_train, Y_train)
train_score = eclf1.score(X_train, Y_train)
print(train_score)

# prediction
predicted = eclf1.predict(X_test)
test_score = eclf1.score(X_test, Y_test)
print(test_score)


