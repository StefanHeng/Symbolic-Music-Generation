# Generated Samples

This directory holds conditionally generated samples based on iconic pop songs/classical music. 



We selected the following songs for generation, you’ll find the corresponding music files in the [`original`](https://github.com/StefanHeng/Symbolic-Music-Generation/blob/master/generated-samples/original) folder. 

Pop 

- *Shape of You* by *Ed Sheeran*, 2017 
- *Rolling in the Deep* by *Adele*, 2011
- *Señorita* by *Camila Cabello*, 2019
- *See You Again* by *Wiz Khalifa* Ft. *Charlie Puth*, 2015
- *告白气球* by *周杰伦*, 2016 
- *走马* by *陈粒*, 2015
- *飘向北方* by *黄明志*, 2016 
- *年少有为* by *李荣浩*, 2018 
- *李白* by *李荣浩*, 2013 
- *平凡之路* by *朴树*, 2014 
- *丑八怪* by *薛之谦*, 2013 
- *演员* by *薛之谦*, 2015 
- *倒数* by *邓紫棋*, 2018 
- *挪威的森林* by *伍佰*, 1996 
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
- *House of the Rising Sun* by *The Animals*, 1964 
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



Below we include some cherry-picked generation samples from model variants. We include a broader set of samples on YouTube. 



You will find all the generated pieces (MXL files) in the remaining folders. You can listen to them with your favorite synthesizer. I use [MuseScore](https://musescore.org). 

<br>





### [22-12_Transformer XL, longer-seq](https://github.com/StefanHeng/Symbolic-Music-Generation/tree/master/generated-samples/22-12_Transformer%20XL%2C%20longer-seq)

Just like the one below, except Transformer XL-small with sequence length 2048 and segment length 1024. 

<br>



[YouTube playlist](https://www.youtube.com/playlist?list=PL1-KLz0i9pBGDRH2zVjHqqBBn2PG2dK3H)

**Top-K sampling with K = 8, given first 8 bars**

*Canon* 

- [![](https://markdown-videos.deta.dev/youtube/4qi42k-YBVg)](https://youtu.be/4qi42k-YBVg)

*Merry Go Round of Life* 

- [![](https://markdown-videos.deta.dev/youtube/2j5BH7k5Ytc)](https://youtu.be/2j5BH7k5Ytc)

*平凡之路* 

- [![](https://markdown-videos.deta.dev/youtube/2WqlsmRiQaE)](https://youtu.be/2WqlsmRiQaE)

*年少有为* 

- [![](https://markdown-videos.deta.dev/youtube/44-yVc_aFzc)](https://youtu.be/44-yVc_aFzc)

<br>





### [22-11_Transformer XL, degree-pitch](https://github.com/StefanHeng/Symbolic-Music-Generation/tree/master/generated-samples/22-11_Transformer%20XL%2C%20degree-pitch)

Sequence length 1024, segment length 512 Transformer XL-base trained on POP909, MAESTRO and the entire Lakh MIDI Datasets for 128 epochs, with examples-proportional mixing, key augmentation and degree pitch. 

<br>



[YouTube playlist](https://www.youtube.com/playlist?list=PL1-KLz0i9pBHOYSONfmgyyLkmrvdTfyZ4)

**Top-K sampling given first 8 bars**

*The Marriage of Figaro* with K = 10 

- [![](https://markdown-videos.deta.dev/youtube/vS3sv37RO30)](https://youtu.be/vS3sv37RO30)

*Shape of You* with K = 8

- [![](https://markdown-videos.deta.dev/youtube/HLzb5I8oNpU)](https://youtu.be/HLzb5I8oNpU)

*Señorita* with K = 8

- [![](https://markdown-videos.deta.dev/youtube/MB4FXnoLcIg)](https://youtu.be/MB4FXnoLcIg)

*告白气球* with K = 16

- [![](https://markdown-videos.deta.dev/youtube/2ACHwYMn-2g)](https://youtu.be/2ACHwYMn-2g)

<br>



### [22-04_Reformer + key-aug](https://github.com/StefanHeng/Symbolic-Music-Generation/tree/master/generated-samples/22-04_Reformer%20%2B%20key-aug)

Just like the one below, except a possible key of each song is inserted with its key (key augmentation). 

<br>





### [22-04_Reformer](https://github.com/StefanHeng/Symbolic-Music-Generation/tree/master/generated-samples/22-04_Reformer)

Sequence length 2048 Reformer-base trained on POP909 and a subset of the cleaned version of Lakh Midi Dataset for 8 epochs, with midi pitch 

<br>

