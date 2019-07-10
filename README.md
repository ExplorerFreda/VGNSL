# [Visually Grounded Neural Syntax Acquisition](https://ttic.uchicago.edu/~freda/paper/shi2019visually.pdf)

[Freda Shi](https://ttic.uchicago.edu/~freda), [Jiayuan Mao](http://jiayuanm.com), 
[Kevin Gimpel](https://ttic.uchicago.edu/~kgimpel), [Karen Livescu](https://ttic.uchicago.edu/~klivescu)

**ACL 2019** &nbsp; 
[[paper]](https://ttic.uchicago.edu/~freda/paper/shi2019visually.pdf) 
[[project page]](https://ttic.uchicago.edu/~freda/project/vgnsl/)
[[bib]](https://ttic.uchicago.edu/~freda/file/bib/shi2019visually.bib)

![model.jpg](https://ttic.uchicago.edu/~freda/project/vgnsl/model.jpg)

## Requirements
PyTorch >= 1.0.1 

See also [env/conda_env.txt](./env/conda_env.txt) for detailed (but maybe not necessary) environment setup. 

## Data Preparation

Download our pre-processed data [here](https://drive.google.com/open?id=1Fpxvcs03Vycg_WaV6Z2UvDvS-2B_LgCu) and unzip it to the data folder. 

To run our demo, the data folder should be organized as follows

```
data
├── mscoco
│   ├── dev_caps.txt
│   ├── dev_ims.npy
│   ├── test_caps.txt
│   ├── test_ground-truth.txt
│   ├── train_caps.txt
│   ├── train_ims.npy
│   └── vocab.pkl
...
```


## Training
VG-NSL
```
bash demos/demo_train.sh
```

VG-NSL+HI
```
bash demos/demo_train_head-initial.sh
```

## Inference/Testing
After training a model, we can test it by running 
```
bash demos/demo_test.sh
```

## Citation 
If you found the codebase useful, please consider citing
```text
@inproceedings{shi2019visually,
    Title = {Visually Grounded Neural Syntax Acquisition},
    Author = {Shi, Haoyue and Mao, Jiayuan and Gimpel, Kevin and Livescu, Karen},
    Booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    Year = {2019}
}
```

## Acknowledgement
The visual-semantic embedding part is adapted from the codebase of [VSE++](https://github.com/fartashf/vsepp) (Faghri et al., BMVC 2018) and [VSE-C](https://github.com/ExplorerFreda/VSE-C) (Shi et al., COLING 2018).
Part of the basic code is adapated from Jiayuan's personal Python toolkits [Jacinle](https://github.com/vacancy/Jacinle/) and Freda's toolkits [Witter](https://github.com/explorerfreda/witter/). 
We also thank [Victor Silva](http://www.victorssilva.com/) for providing the original [concreteness estimation codebase](https://github.com/victorssilva/concreteness) (Hessel et al., NAACL-HLT 2018). 

## License
MIT  
