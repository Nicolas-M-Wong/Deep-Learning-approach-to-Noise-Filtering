############### Autoencoder réduisant le bruit ambient sur un enregistrement d'un cri d'oiseau ###############
import numpy as np 
import pandas as pd
import librosa.display
length = 216

#Les différents dataset sont disponible sur le lien suivant:
    # https://bit.ly/3l8yq4M
# Les données présentent dans le CSV sont répartie dans un seul tableau
# de taille 102400x216, soit 800 matrice de taille 128x216. 
# Pour pouvoir avoir un réseau de neurone qui fonctionne nous devons 
# reconstruire en une matrice de 4 dimensions. Chaque élément de 
# cette matrice est un spectrogramme.

#%% Récupération des données présente dans le CSV

data_set = ['datasetgoelandargente/data-goeland-1sound.csv']
csv_file = pd.read_csv('../input/'+data_set[0],sep=';',header=None)

csv_file = np.array(csv_file)
axis_1,axis_2 = np.shape(csv_file)
images_complete = np.zeros((int(axis_1/128),128,axis_2))

print(np.shape(images_complete))

for i in range (0,int(axis_1/128)):
    sub_matrix = np.zeros ((128,length))
    for k in range (0,128):
        for j in range (0,length):
            sub_matrix[k,j] = float(csv_file [i*128+k,j])
        
    images_complete [i] = sub_matrix
    
print("Récupération des données terminées")

dataset_size = int(axis_1/128)
train_set_size = int(4/5*dataset_size)
test_set_size = dataset_size - train_set_size
train_images = np.zeros ((train_set_size,128,length,1))
test_images = np.zeros ((test_set_size,128,length,1))
train_images[:,:,:,0] = images_complete [0:train_set_size]
test_images [:,:,:,0] = images_complete [train_set_size:dataset_size]
print("Construction de l'ensemble d'entrainement et de test terminé")

#%% Normalisation des données

max_ = [np.amax(train_images),np.amax(test_images)]

max_tot = np.amax(max_)

def normalisation_data (data_set,maximum):
    data_set = data_set.astype('float32') / maximum
    return (data_set)

train_images = normalisation_data(train_images,max_tot)
test_images = normalisation_data(test_images,max_tot)

# Nous cherchons l'élément dont la valeur est la plus grande dans l'ensemble de 
# de nos données. Ensuite nous divisons toutes ces données par cette valeur de 
# telle sorte à avoir que des valeurs comprise entre 0 et 1. Cela nous donnera 
# par la suite un calcul de l'erreur entre 0 et 1, s'il est plus grand que 1
# notre réseau ne fonctionne pas (les poids seront probablement nul dans 
# l'ensemble du réseau)

#%% Création de l'autoencoder et entrainement du réseau

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import librosa
import soundfile as sf


def mel_2_audio (spectrogram,sr,name):
    x = librosa.feature.inverse.mel_to_audio(spectrogram)
    sf.write(name,x,sr)

activation_function = 'relu'
print(np.shape(train_images),np.shape(test_images))

model = models.Sequential()

filter_size = 32

# Cette structure est la taille maximale de l'auto-encoder, vous pouvez améliorer
# sa qualité de reconstruction en réduisant le nombre de couple de couche
# maxpooling et de couche upsampling

model.add(layers.Conv2D(filter_size, (2, 2), activation= activation_function, padding='same',
                        input_shape=(128, length,1)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filter_size, (2, 2), activation=activation_function, padding='same'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filter_size, (2, 2), activation=activation_function, padding='same'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filter_size, (2, 2), activation=activation_function, padding='same'))


model.add(layers.UpSampling2D(input_shape=(2, 2, 1)))
model.add(layers.Conv2D(filter_size, (2, 2), activation=activation_function, padding='same'))


model.add(layers.UpSampling2D(input_shape=(2, 2, 1)))
model.add(layers.Conv2D(filter_size, (2, 2), activation=activation_function, padding='same'))


model.add(layers.UpSampling2D(input_shape=(2, 2, 1)))
model.add(layers.Conv2D(1, (2, 2), activation = activation_function, padding='same'))


model.summary() # Affichage de la structure du réseau

model.compile(optimizer='adam',loss='mse') 

train_history = model.fit(train_images, train_images, epochs=50, 
                    validation_data = (test_images, test_images)) # Entrainement
# du réseau construit ci dessus

# Dans cette structure en utilisant un GPU chaque époque prend entre 1 à 2 
# secondes pour être entrainé

# Affichage de l'évolution de l'erreur au cours de l'entrainement, elle doit 
# convergente pour espérer avoir des résultats généralisable sur un grand nombre
# d'extrait de qualité variable

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
plt.show()

#%% Sélection de données a afficher pour évaluer la qualité de nos résultats

train2 = np.zeros ((train_set_size,128,length,1))
train2[0:11,:,:,0] = images_complete [695:706]
decoded_imgs = model.predict(train2[0:11])
n = 5

#%% Affichage des spectrogrammes en couleur
# L'affichage des spectrogramme et l'enregistrement des fichiers audio peut être
# relativement longue car elle nécessite d'effectuer un spectrogramme puis de 
# l'inverser
def mel_spectrogram_color_gradient (scale,sr,name_graph):
    plt.figure()
    X = librosa.stft(scale)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme MEL '+name_graph)
    plt.tight_layout()
    plt.show()
    
plt.figure(figsize=(20, 4))
nb = 0
for i in range(1, n + 1):
    nb+=1
    name = "corneille des rivages extrait "+str(nb)
    
    # Affichage de l'enregistrement initial sous forme de spectrogramme
    x = librosa.feature.inverse.mel_to_audio(train2[i].reshape(128, length))
    mel_spectrogram_color_gradient(x,22050,name+" original")
    
    # Affichage du spectrogramme produit par l'autoencoder
    x = librosa.feature.inverse.mel_to_audio(decoded_imgs[i].reshape(128, length))
    mel_spectrogram_color_gradient(x,22050,name+" decodé")
    
plt.show()

#%% Sortie sous forme de fichier audio

for i in range(1, n + 1):
    
    # Affichage de l'enregistrement initial sous forme de spectrogramme
    name_test = "test_train_original "+str(i)+".wav"
    spectrogram_1 = train2[i].reshape(128, length)
    mel_2_audio(spectrogram_1,22050,name_test)
    
    # Affichage du spectrogramme produit par l'autoencoder
    name_reconstructed = "test_train_reconstructed "+str(i)+".wav"
    spectrogram_2 = decoded_imgs[i].reshape(128, length)
    mel_2_audio(spectrogram_2,22050,name_reconstructed)
    
#%% Vérification sur un spectrogramme rempli de bruit gaussien exclusivement

def noise_addition(data_set):
    noise_factor = 0.2
    max_power = np.random.randint(0,150)/500
    noisy_data = data_set + noise_factor * np.random.normal(loc=0.0, scale=max_power, size=data_set.shape)
    return (noisy_data)

empty_matrix = np.zeros((1,128,216,1))
empty_matrix = noise_addition(empty_matrix) # Création de la matrice bruité
prediction = model.predict(empty_matrix)

name = "bruit gaussien"

# Affichage de l'enregistrement initial sous forme de spectrogramme
x = librosa.feature.inverse.mel_to_audio(empty_matrix[0].reshape(128, length))
mel_spectrogram_color_gradient(x,22050,name+" original")

# Affichage du spectrogramme produit par l'autoencode
x = librosa.feature.inverse.mel_to_audio(prediction.reshape(128, length))
mel_spectrogram_color_gradient(x,22050,name+" decoded")

# Nos résultat nous permettent de bien vérifier qu'il y a une élimination quasi 
# complète du bruit gaussien lorsqu'il met en entrée de notre réseau déjà entrainé 

