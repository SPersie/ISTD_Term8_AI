import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### Read data into a DataFrame
def read_data():
    images = []
    labels = []
    file = open('trainset_gt_annotations.txt', 'r')
    for line in file:
        a = line.rstrip().split()
        images.append(a[0])
        label = a[1:]
        label = list(map(int, label))
        labels.append(label)
    np_labels = np.array(labels)
    
    cats = []
    file2 = open('concepts_2011.txt', 'r')
    for line in file2:
        a = line.rstrip().split()
        cats.append(a[1])
    cats = cats[1:]
    
    df = pd.DataFrame(np_labels, columns=cats)
    df.insert(loc=0, column='Image', value=images)
    
    return df

def train_test_split(df):
    temp_train_list = []
    temp_validate_list = []
    temp_test_list = []
    headers = ['Image', 'Spring', 'Summer', 'Autumn', 'Winter']
    for i in range(1, 5):   
        temp_df = df.loc[df[headers[i]] == 1]
        temp_train = temp_df.iloc[0:int(temp_df.shape[0]*0.6)]
        temp_train_list.append(temp_train)
        temp_validate = temp_df.iloc[int(temp_df.shape[0]*0.6):int(temp_df.shape[0]*0.7)]
        temp_validate_list.append(temp_validate)
        temp_test = temp_df.iloc[int(temp_df.shape[0]*0.7):int(temp_df.shape[0])]
        temp_test_list.append(temp_test)

    train = pd.concat(temp_train_list, axis=0)
    validate = pd.concat(temp_validate_list, axis=0)
    test = pd.concat(temp_test_list, axis=0)

    train = train.drop_duplicates(subset=['Image'])
    validate = validate.drop_duplicates(subset=['Image'])
    test = test.drop_duplicates(subset=['Image'])
    return train, validate, test


def get_np_array(df):
#     img_index = df.index.tolist()
#     print(img_index)
#     print(img_index)
#     img_names = os.listdir('imageclef2011_feats/')
    img_names = df['Image'].tolist()
    img_labels = pd.concat([df[['Spring']], df[['Summer']], df[['Autumn']], df[['Winter']]], axis=1)
    
    np_img = np.zeros((1, 1024))
    for i in range(len(img_names)):
        temp = np.load('imageclef2011_feats/' + img_names[i] + '_ft.npy')
        re_temp = temp.reshape(1, 1024)
        np_img = np.append(np_img, re_temp, axis=0)
    
    np_img = np.delete(np_img, (0), axis=0)
    np_labels = img_labels.values
    return np_img, np_labels

def first_exp(df):
    first_df = df.loc[(df['Spring'] == 1) | (df['Summer'] == 1) | (df['Autumn'] == 1) | (df['Winter'] == 1)]
    name = first_df[['Image']]
    spring = first_df[['Spring']]
    summer = first_df[['Summer']]
    autumn = first_df[['Autumn']]
    winter = first_df[['Winter']]
    
    opt = pd.concat([name, spring, summer, autumn, winter], axis=1)
    return opt

def get_data():
    df = read_data()
    opt = first_exp(df)
    train, validate, test = train_test_split(opt)
    
    train_x, train_y = get_np_array(train)
    validate_x, validate_y = get_np_array(validate)
    test_x, test_y = get_np_array(test)
    
    return train_x, train_y, validate_x, validate_y, test_x, test_y



### train one time
def train_img(train_x, train_y, c):
    svms = []
    for i in range(4):
#         print(train_x)
#         print(train_y[:,i])
        svm = SVC(kernel='linear', C=c, probability=True)
        svm.fit(train_x, train_y[:,i])
        svms.append(svm)
    return svms

def predict_img(validate_x, validate_y, svms):
    y_preds = []
    for i in range(4):
        y_pred = svms[i].predict_proba(validate_x)
#         print(y_pred)
        y_preds.append(np.delete(y_pred, (0), axis=1))
        
    opt = np.concatenate(y_preds, axis=1)
    
    predictions = []
    ground_truth = []
    for i in range(validate_x.shape[0]):
        predictions.append(np.argmax(opt[i,:]))
        ground_truth.append(np.argmax(validate_y[i,:]))
    
    return predictions, ground_truth


def vanilla_acc(predictiions, ground_truth):
	truth = 0
	for i in range(len(predictiions)):
		if predictiions[i] == ground_truth[i]:
			truth += 1

	acc = truth / len(predictiions)
	return acc

def cls_wise_acc(predictiions, ground_truth):
	truth0 = 0
	cor0 = 0
	truth1 = 0
	cor1 = 0
	truth2 = 0
	cor2 = 0
	truth3 = 0
	cor3 = 0

	for i in range(len(predictiions)):
		if ground_truth[i] == 0:
			truth0 += 1
			if predictiions[i] == 0:
				cor0 += 1

		if ground_truth[i] == 1:
			truth1 += 1
			if predictiions[i] == 1:
				cor1 += 1

		if ground_truth[i] == 2:
			truth2 += 1
			if predictiions[i] == 2:
				cor2 += 1

		if ground_truth[i] == 3:
			truth3 += 1
			if predictiions[i] == 3:
				cor3 += 1

	cls_wise_acc = (cor0/truth0 + cor1/truth1 + cor2/truth2 + cor3/truth3) / 4
	return cls_wise_acc

def select_c(train_x, train_y, validate_x, validate_y):
	best_acc = -1
	best_model = None
	c = [0.01, 0.1, 0.1 **0.5, 1, 10 **0.5, 10, 100 **0.5]
	for i in range(len(c)):
		svms = train_img(train_x, train_y, c[i])
		predictiions, ground_truth = predict_img(validate_x, validate_y, svms)
		acc = cls_wise_acc(predictiions, ground_truth)
		if acc >= best_acc:
			best_acc = acc
			best_model = svms
			best_c = c[i]
	return best_acc, best_model, best_c


def test_set(best_c):
    df = read_data()
    opt = first_exp(df)
    train, validate, test = train_test_split(opt)

    training = pd.concat([train, validate])
    train_x, train_y = get_np_array(training)
    test_x, test_y = get_np_array(test)

    svms = train_img(train_x, train_y, best_c)
    predictions, ground_truth = predict_img(test_x, test_y, svms)
    vanila = vanilla_acc(predictions, ground_truth)
    cls_wise = cls_wise_acc(predictions, ground_truth)
    return vanila, cls_wise


if __name__ == '__main__':
	train_x, train_y, validate_x, validate_y, test_x, test_y = get_data()
	best_acc, best_model, best_c = select_c(train_x, train_y, validate_x, validate_y)
	vanila, cls_wise = test_set(best_c)






