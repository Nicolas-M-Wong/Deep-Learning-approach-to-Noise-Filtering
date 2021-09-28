# Deep-Learning-approach-to-Noise-Filtering

The aim of this program is to develop a deep autoencoder (DAE) for denoising bird song to improve the quality of existing sound and reduce manpower needed to analyse in situ continuous recording which are becoming more and more common in ornithology. This solution is also an answer to potential low reliability in automated bird song recognition based on machine learning based algorithm.

Our DAE is based on a convolutional network (CNN) which analyse Mel-Spectrogram created with 5 second extract. With the actual dataset, the size reduction of the Mel-Spectrogram depends on the species studied but it is usually not higher than a 2-time reduction. Under this resolution, our network can’t properly reproduce a sound. However, it can be still used relatively accurately for the detection of the species trained on.

The dataset is using sound from the Xeno-Canto project which collaborative program that collect birdsong recording and that are then identify by the community. Our initial data base is using only one extract that is then put inside a random gaussian noise and the position of the sound is random. However, this random position should not bother the network as it is based on CNN which are time invariant.

This project has been developed as part of PMI (Master’s project for IPSA engineering School in Paris) with Simon Cocagne, Valentin Forite and my self Nicolas Wong. We would like also to thanks Mr Omar Al Hammal for his tutoring and help through out this project. 

*The code contains comments in French, a following version will include the translated version of these comments in English.*

*The full report with result explanation and research of the state of the art is also in French but here are some qucik example of expected result*

Result obtained using the python code:

This first example is based on the song an europpean herring gull (Goéland Argenté in French). This song is very specific and usually repeated multiple time few seconds apart from each other. It has a very clear pattern which makes it relatively easy for our model to be trained with it. Compared to the next example, the frequency amplitude is low and the compression rate achievable with acceptable result is higher degree of denoising.

<div align="center">
   
| Original Extract  |  Reconstructed after 2 maxpooling layers |
| ------------- | ------------- |
| <img src="https://github.com/Nicolas-M-Wong/Deep-Learning-approach-to-Noise-Filtering/blob/main/Result/European%20herring%20gull%20-%202%20Maxpooling%20Layer%20-%20Original%20extract.png" width="400"/> | <img src="https://github.com/Nicolas-M-Wong/Deep-Learning-approach-to-Noise-Filtering/blob/main/Result/European%20herring%20gull%20-%202%20Maxpooling%20Layer%20-%20Reconstructed%20Extract.png" width="400"/> |
</div>

American Fish crow

<div align="center">
| Original Extract  |  Reconstructed after 1 maxpooling layers |
| ------------- | ------------- |
| <img src="https://github.com/Nicolas-M-Wong/Deep-Learning-approach-to-Noise-Filtering/blob/main/Result/Fish%20crow%20-%201%20Maxpooling%20Layer%20-%20Original%20Extract.pngg" width="400"/> | <img src="https://github.com/Nicolas-M-Wong/Deep-Learning-approach-to-Noise-Filtering/blob/main/Result/Fish%20crow%20-%201%20Maxpooling%20Layer%20-%20Reconstructed%20Extract.png" width="400"/> |
   
