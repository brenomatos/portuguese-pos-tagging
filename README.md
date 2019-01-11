# Part-Of-Speech Tagging (Portuguese)

Code regarding the second practical assignment from Natural Language Processing class @ UFMG.
The assignment was to perform a POS-Tag task on a portuguese corpus.
## Files Needed

In this assignment, I used the Mac-Morpho dataset. Mac-Morpho is a corpus of Brazilian Portuguese texts annotated with part-of-speech tags. It's files and manual (in portuguese) can be found [here](http://nilc.icmc.usp.br/macmorpho/).


## Running the code
First, clone this repository.

Then, execute the command:

```shell
python3 main.py
```

Note that you can change the # of epochs, batch size and the sliding window size in the main method:
```python3
def main(window_size,epochs,batch_size)
```
The results will be stored in the 'results' folder.

## Results
For each model created, there will be 2 results files: one is named like 'x-y.csv', being 'x' the sliding window size, and
'y' being the # of epochs. The other file is named like 'total_accuracy_x-y.txt',
with 'x' and 'y' having the same meaning as in the previous file.

The first file contains the accuracy of the model regarding every grammatical class considered. The second file contains the accuracy of the model as a whole, considering total hits/misses.

### Graphs
This code will also create graphs for each model created, stored in the 'results/graphs/' folder. The graph displays each class's accuracy like the example below:
![Example Graph](/results/graphs/accuracy_by_class_3-1.png "Example Graph")

## Built With
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/stable/)

## Contributing
Pull requests are welcome!

## Acknowledgments
Some methods from 'graphs.py' were inspired by my friend and brazillian developer [Rafael](https://rafaatsouza.github.io/). Check his implementation for a portuguese POS-Tagger [here](https://github.com/rafaatsouza/nlp_tp2)

## License
[MIT](https://choosealicense.com/licenses/mit/)
