# Symbolic Music Generation
Symbolic music generation taking inspiration from NLP Language Modeling, human composition process and music theory. 

Contributor: [Stefan/Yuzhao Heng](https://stefanheng.github.io), [Carson/Jiachun Zhang](https://github.com/SonyaInSiberia), [Xiaoyang Song](https://github.com/Xiaoyang-Song).
Mentored by [Artem Abzaliev](http://artem.site44.com), 
supported by [Prof. Rada Mihalcea](https://web.eecs.umich.edu/~mihalcea/) at [LIT](https://lit.eecs.umich.edu/people.html).






## Run the Scripts

Python version `3.10.8`. 

Modify the `DIR_DSET` variable in file [`data_path.py`](https://github.com/StefanHeng/Symbolic-Music-Generation/blob/master/musicnlp/util/data_path.py) 
as instructed.


Run 
```bash
python musicnlp/util/config.py
```

A folder named as `DIR_DSET` should be kept at the same level as 
this repository, with dataset folder names specified as 
in [`config.json`](https://github.com/StefanHeng/Symbolic-Music-Generation/blob/master/musicnlp/util/config.json).

Add datasets to `DIR_DSET`, then see [`music_export.py`](https://github.com/StefanHeng/Symbolic-Music-Generation/blob/master/musicnlp/preprocess/music_export.py) for encoding Music MXL files. 






## Music Samples 
See [`generated-samples`](https://github.com/StefanHeng/Symbolic-Music-Generation/tree/master/generated-samples). 





## Resources 

Datasets linked [here](https://drive.google.com/drive/folders/1T4TDRjj4nwQWRoytBYNg5VlqHCivTjyP?usp=sharing). 

Some visualizations and training log [here](https://drive.google.com/drive/folders/1hbnX2WkrikgoL7gVM-bMF29tjlJfllu0?usp=sharing). 

