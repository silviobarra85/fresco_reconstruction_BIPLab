# Fresco Reconstruction (BIPLAB)

For each set of fragments, the reconstructed fresco is generatedl.
The reference dataset are:
- [Digital Anastylosis of Frescoes challeNgE (DAFNE) Dataset ANASTYLOSIS DB1 - for development](https://vision.unipv.it/DAFchallenge/ANASTYLOSIS%20DB1/dataset_ANASTYLOSIS-DB1.html)
- [Digital Anastylosis of Frescoes challeNgE (DAFNE) Dataset ANASTYLOSIS DB2 - for testing](https://vision.unipv.it/DAFchallenge/ANASTYLOSIS%20DB2/dataset_ANASTYLOSIS-DB2.html)

 

If you're using **Pycharm**, after clone the repo use this doc to create the venv in the repository: [https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html), use `./venv/` for the folder name, because it's added on `.gitignore`.

After that, you can easly load the dependencies using: 

Remember: in order to use Tensorflow, Tensorflow-GPU you must use Python 3.6 with Pip at latest version.

Also with pycharm there are some conflict with pip. This can be a solution (after create venv)
```
python -m pip install --upgrade pip

pip install -r venv-configuration.txt
```

If you install other libs, remember to export the new configuration with the following comand: 

```
pip freeze > venv-configuration.txt
```

## Execution
For executing the project, simply

```

python dafne.py -d <dataset_folder> -l <list of fresco folders space-separated> -o <output_folder>
 

```

The file searchs for folders properly named inside the data folder.
Each folder has a name like: YY_XXXXXXXX, in which YY is a progressive number and XXXX..XXX is the name of the folder.
XXXXXXX is also the name of the fresco with jpg extension.
The folder must contain a frag_eroded folder which contains a set of fragments named:
- frag_eroded_1
- frag_eroded_2
- frag_eroded_3
...
- frag_eroded_n

#Output
A SOLUTION folder is created, containing a reconstructed image (with .png extension) for each reconstructed fresco/set of fragments.
Also, a txt file containing list of fragments positions and rotation degree is returned.


## Note
Some warning may appear during execution.


## Links

- [Configuring Virtualenv Environment Pycharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html)


## Author

- Paola Barra (@Paps90)
- Silvio Barra (@silviobarra85)
- Fabio Narducci
