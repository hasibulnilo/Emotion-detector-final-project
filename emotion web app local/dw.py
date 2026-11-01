import urllib.request

url = "https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-face-generation/gender_model.h5"
urllib.request.urlretrieve(url, "gender.h5")

print("Gender model downloaded")