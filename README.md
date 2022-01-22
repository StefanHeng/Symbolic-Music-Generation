# Symbolic Music Generation
Symbolic music generation taking inspiration from NLP and human composition process. 

Contributor: [Stefan/Yuzhao Heng](https://stefanheng.github.io), [Carson/Jiachun Zhang](https://github.com/SonyaInSiberia).
Mentored by [Artem Abzaliev](http://artem.site44.com) 
at [LIT](https://lit.eecs.umich.edu/people.html).


## To use 

Create a file `data_path.py` in root level. 

In the file specify the following variables with 
your system data path, and relative repository & dataset folder names:
```python
PATH_BASE = '/Users/stefanh/Documents/UMich/Research/Music with NLP'  # System data path
DIR_PROJ = 'Symbolic_Music_Generation'  # Repo root folder name 
DIR_DSET = 'datasets'  # Dataset root folder name 
```

Run 
```bash
$ python config.py
```


Also, a `datasets` folder should be kept at the same level as 
this repository, with dataset folder names specified as 
in [`config.json`](https://github.com/StefanHeng/Symbolic-Music-Generation/blob/master/config.json).

