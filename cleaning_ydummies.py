import json
import numpy as np
import pandas as pd




####################### Data Cleaning
#https://github.com/ahegel/yelp-dataset/blob/master/Predicting%20Star%20Ratings.ipynb
#load and preview the data
review_data = []
with open('business_sample.json', encoding="utf8") as f:
    for line in f:
        review_data.append(json.loads(line))

review_df = pd.DataFrame.from_dict(review_data)
print(review_df.head())
print(review_df.shape) 
review_df = review_df[0:5000] # for trying, limit the dataset
# print(review_df.head())


 
#[5]
review_df['categories_clean'] = review_df['categories']
# clean 'categories' and use them as features
categories_df = review_df.categories_clean.str.get_dummies(sep=',')
# merge
review_df = review_df.merge(categories_df, left_index=True, right_index=True)
# remove intermediate columns (no longer needed)
review_df.drop(['categories', 'categories_clean'], axis=1, inplace=True)
print(review_df.head())


# making rating dummy - rating is discrete
 # use stars as features
 #https://towardsdatascience.com/the-dummys-guide-to-creating-dummy-variables-f21faddb1d40
ydummies = pd.get_dummies(review_df['stars'])
print(ydummies.head())
review_df = pd.concat([review_df, ydummies], axis = 1)
print(review_df.head())
review_df = review_df.merge(ydummies, left_index= True, right_index = True)
print(review_df.head())


review_df["stars_cat"] = review_df['stars'].astype(str)
# review_df["stars_cat"]= str(review_df['stars'])


#one-hot encode the business attributes and their sub-attributes, which are currently in dictionaries in the 'attributes' column.

#[6]
print(review_df['attributes'].head())


#[7]

review_df = review_df.join(pd.DataFrame(review_df['attributes'].to_dict()).T)
# further split sub-attributes into their own columns
# cols_to_split = ['BusinessParking', 'Ambience', 'BestNights', 'GoodForMeal', 'HairSpecializesIn', 'Music']
# for col_to_split in cols_to_split:
#     new_df = pd.DataFrame(review_df[col_to_split].to_dict(), index=[0]).T
#     ############# index problem 
#     new_df.columns = [col_to_split + '_' + str(col) for col in new_df.columns]
#     review_df = review_df.join(new_df)

# # for col_to_split in cols_to_split:
# #     df1 = pd.DataFrame(review_df[col_to_split].values.tolist())
# #     df1.columns = col_to_split + '_' + df1.columns
# #     review_df = review_df.join(df1.columns)

# review_df.drop(['attributes'] + cols_to_split, axis=1, inplace=True)
# print(review_df.head())

#[8]
# columns with non-boolean categorical values:
cols_to_split = ['AgesAllowed', 'Alcohol', 'BYOBCorkage', 'NoiseLevel', 'RestaurantsAttire', 'Smoking', 'WiFi']
new_cat = pd.concat([pd.get_dummies(review_df[col], prefix=col, prefix_sep='_') for col in cols_to_split], axis=1)

# keep all columns (not n-1) because 0's for all of them indicates that the data was missing (useful info)
review_df = pd.concat([review_df, new_cat], axis=1)
review_df.drop(cols_to_split, inplace=True, axis=1)
review_df.head()




#[9]
# convert true/false columns to 0/.5/1 for false/missing/true
nonb = ['BikeParking', 'BusinessAcceptsCreditCards', 'GoodForKids', 'HappyHour', 'OutdoorSeating', 'Caters', 'RestaurantsReservations', 'RestaurantsTakeOut', 'RestaurantsDelivery', 'HasTV', 'ByAppointmentOnly', 'RestaurantsGoodForGroups', 'WheelchairAccessible','DogsAllowed','CoatCheck', 'DriveThru', 'RestaurantsTableService', 'GoodForDancing', 'BusinessAcceptsBitcoin', 'AcceptsInsurance']
# print(review_df[nonb].head(20))
if nonb:
	for i in nonb:
		d = {'True': True, 'False': False}
		review_df[i]= review_df[i].map(d)
		#https://stackoverflow.com/questions/17702272/convert-pandas-series-containing-string-to-boolean
		review_df = review_df.fillna(0.5).apply(pd.to_numeric, errors='ignore')
else:
	print(i + 'does not exist')
# print(review_df[nonb].head(20))



# deal with missing values in postal code
# print(review_df['postal_code'].isnull().sum())
review_df['postal_code'] = review_df['postal_code'].fillna(0)
# print(review_df['postal_code'].isnull().sum())

# check that all nulls are removed
# print(review_df.isnull().sum().sum())
############# It should be zero after step 9, 10



# dropping unnecessary variables


cols = list(review_df)
# cols.insert(0, cols.pop(cols.index('BusinessParking_0')))
# cols.insert(0, cols.pop(cols.index('Ambience_0')))
# cols.insert(0, cols.pop(cols.index('BestNights_0')))
# cols.insert(0, cols.pop(cols.index('GoodForMeal_0')))
# cols.insert(0, cols.pop(cols.index('HairSpecializesIn_0')))
# cols.insert(0, cols.pop(cols.index('Music_0')))
cols.insert(0, cols.pop(cols.index('address')))
cols.insert(0, cols.pop(cols.index('state')))
cols.insert(0, cols.pop(cols.index('city')))
cols.insert(0, cols.pop(cols.index('latitude')))
cols.insert(0, cols.pop(cols.index('longitude')))
cols.insert(0, cols.pop(cols.index('review_count')))
cols.insert(0, cols.pop(cols.index('is_open')))
cols.insert(0, cols.pop(cols.index('hours')))
# cols.insert(0, cols.pop(cols.index('BYOB')))
#the below data is not in X
# 'AgesAllowed', 'BYOBCorkage', 'Smoking',
# 'HappyHour', 'WheelchairAccessible', 'DogsAllowed','CoatCheck', 'DriveThru', 'GoodForDancing',
# print(cols)
review_df = review_df.loc[:, cols]

# review_df.drop(review_df.columns[0:10], axis=1, inplace = True)


# moving stars to the first column
# cols = list(review_df)
# ['name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories', 'hours', 'BikeParking', 'BusinessAcceptsCreditCards', 'GoodForKids', 'HappyHour', 'OutdoorSeating', 'Caters', 'RestaurantsReservations', 'RestaurantsTakeOut', 'RestaurantsDelivery', 'RestaurantsGoodForGroups', 'HasTV', 'RestaurantsPriceRange2', 'ByAppointmentOnly', 'WheelchairAccessible', 'DogsAllowed', 'CoatCheck', 'RestaurantsTableService', 'BusinessAcceptsBitcoin', 'GoodForDancing', 'AcceptsInsurance', 'DriveThru', 'Corkage', 'BYOB', 'BusinessParking_0', 'Ambience_0', 'BestNights_0', 'GoodForMeal_0', 'HairSpecializesIn_0', 'Music_0', "AgesAllowed_u'21plus'", "AgesAllowed_u'allages'", "Alcohol_'beer_and_wine'", "Alcohol_'full_bar'", "Alcohol_'none'", "Alcohol_u'beer_and_wine'", "Alcohol_u'full_bar'", "Alcohol_u'none'", "BYOBCorkage_'yes_corkage'", "BYOBCorkage_'yes_free'", "NoiseLevel_'average'", "NoiseLevel_'quiet'", "NoiseLevel_'very_loud'", "NoiseLevel_u'average'", "NoiseLevel_u'loud'", "NoiseLevel_u'quiet'", "NoiseLevel_u'very_loud'", "RestaurantsAttire_'casual'", "RestaurantsAttire_'dressy'", "RestaurantsAttire_u'casual'", "Smoking_u'no'", "Smoking_u'outdoor'", "Smoking_u'yes'", "WiFi_'free'", "WiFi_'no'", "WiFi_u'free'", "WiFi_u'no'"]
# print(cols)
cols.insert(0, cols.pop(cols.index('stars')))


# print(cols)
review_df = review_df.loc[:, cols]
# print(list(review_df))

# set the variable order 




cols = list(review_df)
cols.insert(0, cols.pop(cols.index('stars_cat')))
cols.insert(0, cols.pop(cols.index('5.0_y')))
cols.insert(0, cols.pop(cols.index('4.5_y')))
cols.insert(0, cols.pop(cols.index('4.0_y')))
cols.insert(0, cols.pop(cols.index('3.5_y')))
cols.insert(0, cols.pop(cols.index('3.0_y')))
cols.insert(0, cols.pop(cols.index('2.5_y')))
cols.insert(0, cols.pop(cols.index('2.0_y')))
cols.insert(0, cols.pop(cols.index('1.5_y')))
cols.insert(0, cols.pop(cols.index('1.0_y')))
cols.insert(0, cols.pop(cols.index('stars')))
cols.insert(0, cols.pop(cols.index('GoodForKids')))
cols.insert(0, cols.pop(cols.index('RestaurantsReservations')))
cols.insert(0, cols.pop(cols.index('Caters')))
cols.insert(0, cols.pop(cols.index('RestaurantsTableService')))
cols.insert(0, cols.pop(cols.index('RestaurantsTakeOut')))
# cols.insert(0, cols.pop(cols.index('RestaurantsPriceRange2')))
cols.insert(0, cols.pop(cols.index('OutdoorSeating')))
cols.insert(0, cols.pop(cols.index('BikeParking')))
cols.insert(0, cols.pop(cols.index('HasTV')))
cols.insert(0, cols.pop(cols.index('RestaurantsGoodForGroups')))
cols.insert(0, cols.pop(cols.index('RestaurantsDelivery')))
cols.insert(0, cols.pop(cols.index('BusinessAcceptsCreditCards')))
cols.insert(0, cols.pop(cols.index('BusinessAcceptsBitcoin')))
cols.insert(0, cols.pop(cols.index('ByAppointmentOnly')))
# cols.insert(0, cols.pop(cols.index('Alcohol_u'beer_and_wine'')))
# cols.insert(0, cols.pop(cols.index('NoiseLevel_u'average'')))
# cols.insert(0, cols.pop(cols.index('RestaurantsAttire_'casual'')))
# cols.insert(0, cols.pop(cols.index('RestaurantsAttire_u'casual'')))
# cols.insert(0, cols.pop(cols.index('WiFi_u'no'')))

# print(cols)
review_df = review_df.loc[:, cols]
print(list(review_df))


review_df.to_csv("cleaned_data_ydummies.csv")







