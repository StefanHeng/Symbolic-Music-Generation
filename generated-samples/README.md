# Generated Samples

This directory holds conditionally generated samples based on iconic pop songs/classical music. 



We selected the following songs for generation, you’ll find the corresponding music files in the [`original`](https://github.com/StefanHeng/Symbolic-Music-Generation/blob/master/generated-samples/original) folder. 

Pop 

- *Shape of You* by *Ed Sheeran*, 2017 
- *Rolling in the Deep* by *Adele*, 2011
- *Señorita* by *Camila Cabello*, 2019
- *See You Again* by *Wiz Khalifa* Ft. *Charlie Puth*, 2015
- *平凡之路* by *朴树*, 2014 
- *Sugar* by *Maroon 5*, 2014
- *Faded* by *Alan Walker*, 2015
- *My Heart Will Go On* by *Celine Dion*, 1997
- *Take Me Home Country Roads* by *John Denver*, 1971 
- *Perfect* by *Ed Sheeran*, 2017 
- *Despacito* by *Luis Fonsi* ft. *Daddy Yankee*, 2017 
- *Careless Whisper* by *George Michael*, 1984 
- *Stayin’ Alive* by *Bee Gees*, 1977 
- *Something Just Like This* by *The Chainsmokers* & *Coldplay*, 2018 
- *Beat It* by *Michael Jackson*, 1982
- *Autumn Leaves* by *Frank Sinatra*, 1955



Tunes 

- *Merry Go Round of Life* by *Joe Hisaishi*, 2004  
- *Merry Christmas Mr. Lawrence* by *Ryuichi Sakamoto*, 1983 



Classical 

- *Canon in D* by *Johann Pachelbel*, 1680   
- *Piano Sonata No. 11* by *Mozart*, 1783 
- *Carmen Havanaise* by *Bizet*, 1875 
- *Symphony No.5 in C minor* by *Beethoven*, 1808 
- *Serenade No. 13 for strings in G major* by *Mozart*, 1787
- *Sonata for two Pianos in D major, 375a, K. 448* by *Mozart*, 1781
- Overture from *The Marriage of Figaro* by *Mozart*, 1786
- Overture from *William Tell* by *Rossini*, 1829
- *Flower Duet* from *Lakmé* by *Delibes*, 1883
- *Hallelujah!* from *Messiah* by *George Frideric Handel*, 1741 
- *Für Elise* by *Beethoven*, 1810
- *Ave Maria* by *Charles Gounod*, 1853 
- *Ode to Joy* from *Symphony 9* by *Beethoven*, 1824
- *Moonlight* from *Sonata No. 14 in C-sharp minor* by *Beethoven*, 1801



We [extract music representation](https://github.com/StefanHeng/Symbolic-Music-Generation/tree/master/musicnlp/preprocess/music_extractor.py) from the original songs and feed the first few bars as prompt to the models. The extracted representation are stored as music files in the [`extracted`](https://github.com/StefanHeng/Symbolic-Music-Generation/tree/master/generated-samples/extracted) folder. 



You will find all the generated pieces in the remaining folders. 



Below we include some cherry-picked generation samples from model variants. 





<br>





### Reformer + key-aug, 22-04

Just like the one below, except a possible key of each song is inserted with its key. 





<br>





### Reformer, 22-04

Reformer trained on POP909 and a subset of the cleaned version of Lakh Midi Dataset for 8 epochs, with midi pitch 





<br>

