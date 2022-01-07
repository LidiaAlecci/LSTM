# LSTM

An LSTM character-level language model for predicting fables using PyTorch.

## Pre-processing

In this code I will use “Aesop’s Fables (A Version for Young Readers)” and for simplicity no pre-processing is applied.

Anyway, there are plenty of pre-processing techniques that can be applied to text, I will just list some of them: removing punctuations, lower casing, removing stop words, lemmatizing, removing contractions.

## Layers
This model has the following layers:
- 1 x Embedding layer
- 1 x LSTM layer
- 1 x Fully connected layer

## Model
The first goal is to train a model that predicts fables and try to understand how different hyperparameters effects the final model.
However, to understand how this type of model behaves with other task a second model is trained and evaluated with the task of predicting JSON objects, the function that takes care of this try is 'run_model2'.

## Train
I created a Trainer class that takes care of the training and validation steps, plus after each epoch will try to make two predictions (a greedy and a sampling one) based on one prompt (`A KID coming home ` or `{"id":41968,"name":"runtime exception` in case of the second model).

## Evaluation
To understand the prediction power of the first model, I tried four different predictions:
- A title of a fable which exists in the book: `A FOX AND A CRAB `.
- An invented title, that is not in the book, but it is similar in the style: `THE CAT ON THE TABLE `.
- A little text with similar style: `A hawk flew high in the sky when suddenly `.
- Anything that can be interested: `IN the year 1878 I took my degree of Doctor of Medicine of the University of London` (Incipit of Sherlock: A study in Pink by Conan Doyle).

## Main
- `text_data_properties` is a function that print some properties of the data.
- `run_model` is the function that implements and runs the model that predicts fables as describe above.
- `predict` is a function that based on the model, a vocabulary and a prompt will provide a prediction.
- `run_model2` is the function that implements and runs the model that predicts JSON objects as describe above.

## Results
The training phase is considered done after the model achieves a perplexity below `1.03` which happens after around `10` minutes on a GPU. 

### Fables Model
#### Changing the learning rate (`0.0001` and `0.01`)
Decreasing the learning rate will result in more necessary epochs in order to obtain a good
perplexity value. In particular, a perplexity value below `1.03` was obtained after `30` epochs
in the model with learning rate `0.001`, instead with learning rate `0.0001` it was obtained
after `105` epochs. By increasing the learning rate the model has trouble to converge, in
fact even after `300` epochs the model didn’t reach an optimal perplexity value. This is
because due to a high learning rate the model starts to jump too fast without reaching the
minimum. At the contrary when the learning rate is too low the model makes little steps
and so the number of epochs increases.

By increasing the bptt length to `128` the number of epochs necessary to reach a good
model is the same (`30`) but the model is slightly fast than the other (`10` minutes against `12`
minutes), since we are handling more character in a single batch. By decreasing it (bptt
span=`32`) the model struggle to converge, and even after `300` epochs the model didn’t reach
an optimal perplexity value. I guess this is because a too low bptt span truncated also the
ability of the model to learn.
#### Prediction
- With the title `A FOX AND A CRAB` the prediction was: `of his wool, sheared him
so closely as to cut his skin.
The Squirrel takes a look at them—he can do no more. At one time he is called away;
at another, even dragged off in the Lion’s service.
But see! At last the Squirrel has grown old and become tiresome to the Lion. It is
time for him to retire.`.
- With the title I invented `THE CAT ON THE TABLE` the prediction was: `TREE
A PEASANT had in his garden an Apple Tree which for year after year had borne no
fruit, but served only as a harbor for sparrows and grasshoppers still found shelter in
it. They could not forget, however, that the man had only saved their home for them
from self-interest.
JUPITER, NEPTUNE`.
- With some text `A hawk flew high in the sky when suddenly` the prediction was:
`about on the edge of the rock, as safe from falling as she was from the greedy Wolf
with his false care for her.
[Illustration]
THE HEN AND THE SWALLOW
A HEN who had no nest of her own found some eggs, and, in the kindness of her
heart, thought she would take care of them, and keep them warm.`.
- With the incipit of "A study in scarlet" by Conan Doyle: `IN the year 1878 I took
my degree of Doctor of Medicine of the University of London` and the prediction was:
`s in a house. Listening, he heard the Nurse say, “Stop crying this minute, or I will
throw you out of doors to the Wolf.” The Wolf sat down near the door, thinking within himself, “I shall soon have a good
supper.”
THE WOLF AND THE GOAT
A FOX who had fallen into a deep well was casting about `.

In my opinion, the model sometimes is quite good other a little bit less, e.g., by using
the incipit of "A study in scarlet" the text generated has sense since based on the prompt
(degree of Doctor) it talks then about a nurse that mentions a wolf and so then there is the
wolf’s thinking. With the prompt `A FOX AND A CRAB` the model starts to talk about
two other animals, a squirrel and a lion; similar thing happens with the prompt `THE
CAT ON THE TABLE` where the model arbitrarily had added `TREE` in the title and
then started to talk about this apple tree.

I think overall the results are good even if I’m far from saying that the model is capable to
produce novel texts as humans or even enough close. Anyway, some predicted are equals to
some in the training set, e.g. `The Wolf sat down near the door, thinking within himself,
“I shall soon have a good supper."`.

#### Sampling vs Greedy
By using the sampling algorithm I obtained the following results:
- With the title `A FOX AND A CRAB` the prediction was: `of LyCiff calls not be
room.
Do not the enemy of the same punishment, and each time a little distance below.
Having made up his mind to seize the Lamb, the Swallow would have nothing to do
with him. And to this day the Bat seems ashamed to show himself in the tree, he
said, “My dear Madam, what a b`.
- With the title I invented `THE CAT ON THE TABLE` the prediction was: `Ty The
Two Pots 73 The Goat and the Owlen 196 Jusictive warm Bat espate 180 The Oold
Hound By Haren.
Didges of the Partridge; 20 The Camel 170 Mice made and the Wolf 7`.

As we can see with the sampling algorithm the phrase starts to have less meaning even if
at least the prediction is braver so there is no phrase copy and paste from the training. I
found quite interesting that with the second prompt the model predicted the index of the
book.

### Tv Series Prediction
I chose to train the model on a json file that contains a dataset of tv series, I used the same
code I used before, but I select the other file `TvSeries.txt` and I ran again the model also
by changing the bptt value and the learning rate. This new model achieves a perplexity
under `1.03`:
- after `97` epochs and `32` minutes with a learning rate of `0.001` and a bptt of `64`.
- after `148` epochs and `47` minutes with a learning rate of `0.0001` and a bptt of `64`.
- with a learning rate of `0.01` and a bptt of `64` the model doesn’t converge before `300`
epochs, as in the previous model.
- after `88` epochs and `26` minutes with a learning rate of `0.001` and a bptt of `128`.
- with a learning rate of `0.001` and a bptt of `32` the model doesn’t converge before `300`
epochs, as in the previous model.

I tried to predict the next 300 characters based on the following prompt `{"id":1920,"name"
:"Covid19` for both the different algorithms (sampling and greedy):
- Greedy: `The Haiter Iourle Moniter","permalink":"moon-and-son","start_date":"Ju
n/1992","end_date":"Ju n/1992","country":"UK","network":"BBC one","status":"
Canceled/Ended","image_thumbnail_path":"https://static.episodate.com/images/tv
-show/thumbnail/3755.jpg"}]} {"total":"1000","page":8,"pages":50,"tv_shows":`.
- Sampling: ` The Haiter Id Mavicing","permalink":"Gomedy-on-the-road","start_date
":"1991","end_date":"1995","country":"US","network":"A&E","status":"Canceled/E
nded","image_thumbnail_path":"https://static.episodate.com/images/tv-show/thum
bnail/984.jpg"},{"id":151,"name":"Space Cadets (1997)","permalink":"space-c `.