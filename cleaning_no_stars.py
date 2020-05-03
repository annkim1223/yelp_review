import json
import numpy as np
import pandas as pd


review_data = []
with open('business_no_stars_review.json', encoding="utf8") as f:
    for line in f:
        review_data.append(json.loads(line))

review_df = pd.DataFrame.from_dict(review_data)
print(review_df.head())

print(review_df['attributes'].head())

review_df = review_df.join(pd.DataFrame(review_df['attributes'].to_dict()).T)
# further split sub-attributes into their own columns
# cols_to_split = ['BusinessParking', 'Ambience', 'GoodForMeal']
# #'BestNights','HairSpecializesIn', 'Music'
# for col_to_split in cols_to_split:
#     new_df = pd.DataFrame(review_df[col_to_split].to_dict(), index=[0]).T
#     ############# index problem 
#     new_df.columns = [col_to_split + '_' + str(col) for col in new_df.columns]
#     review_df = review_df.join(new_df)

# review_df.drop(['attributes'] + cols_to_split, axis=1, inplace=True)
# print(review_df.head())


cols_to_split = ['Alcohol',  'NoiseLevel', 'RestaurantsAttire',  'WiFi']
new_cat = pd.concat([pd.get_dummies(review_df[col], prefix=col, prefix_sep='_') for col in cols_to_split], axis=1)

# keep all columns (not n-1) because 0's for all of them indicates that the data was missing (useful info)
review_df = pd.concat([review_df, new_cat], axis=1)
review_df.drop(cols_to_split, inplace=True, axis=1)
review_df.head()




#[9]
# convert true/false columns to 0/.5/1 for false/missing/true
nonb = ['BikeParking', 'BusinessAcceptsCreditCards', 'GoodForKids',  'OutdoorSeating', 'Caters', 'RestaurantsReservations', 'RestaurantsTakeOut', 'RestaurantsDelivery', 'HasTV', 'ByAppointmentOnly', 'RestaurantsGoodForGroups',  'RestaurantsTableService',  'BusinessAcceptsBitcoin', 'AcceptsInsurance']
# 'HappyHour', 'WheelchairAccessible', 'DogsAllowed','CoatCheck', 'DriveThru', 'GoodForDancing',
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

review_df.drop(review_df.columns[0:5], axis=1, inplace = True)



# set the variable order 

# print(list(review_df))

cols = list(review_df)
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


# review_df.drop(review_df.columns[0:11], axis=1, inplace = True)


# moving stars to the first column
# cols = list(review_df)
# # ['name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories', 'hours', 'BikeParking', 'BusinessAcceptsCreditCards', 'GoodForKids', 'HappyHour', 'OutdoorSeating', 'Caters', 'RestaurantsReservations', 'RestaurantsTakeOut', 'RestaurantsDelivery', 'RestaurantsGoodForGroups', 'HasTV', 'RestaurantsPriceRange2', 'ByAppointmentOnly', 'WheelchairAccessible', 'DogsAllowed', 'CoatCheck', 'RestaurantsTableService', 'BusinessAcceptsBitcoin', 'GoodForDancing', 'AcceptsInsurance', 'DriveThru', 'Corkage', 'BYOB', 'BusinessParking_0', 'Ambience_0', 'BestNights_0', 'GoodForMeal_0', 'HairSpecializesIn_0', 'Music_0', "AgesAllowed_u'21plus'", "AgesAllowed_u'allages'", "Alcohol_'beer_and_wine'", "Alcohol_'full_bar'", "Alcohol_'none'", "Alcohol_u'beer_and_wine'", "Alcohol_u'full_bar'", "Alcohol_u'none'", "BYOBCorkage_'yes_corkage'", "BYOBCorkage_'yes_free'", "NoiseLevel_'average'", "NoiseLevel_'quiet'", "NoiseLevel_'very_loud'", "NoiseLevel_u'average'", "NoiseLevel_u'loud'", "NoiseLevel_u'quiet'", "NoiseLevel_u'very_loud'", "RestaurantsAttire_'casual'", "RestaurantsAttire_'dressy'", "RestaurantsAttire_u'casual'", "Smoking_u'no'", "Smoking_u'outdoor'", "Smoking_u'yes'", "WiFi_'free'", "WiFi_'no'", "WiFi_u'free'", "WiFi_u'no'"]
# # print(cols)
# cols.insert(0, cols.pop(cols.index('stars')))
# # print(cols)
# review_df = review_df.loc[:, cols]
print(list(review_df))



review_df.to_csv("cleaned_no_stars.csv")







