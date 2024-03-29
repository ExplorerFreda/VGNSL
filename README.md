# [Visually Grounded Neural Syntax Acquisition](https://ttic.uchicago.edu/~freda/paper/shi2019visually.pdf)

[Freda Shi](https://ttic.uchicago.edu/~freda)\*, [Jiayuan Mao](http://jiayuanm.com)\*, 
[Kevin Gimpel](https://ttic.uchicago.edu/~kgimpel), [Karen Livescu](https://ttic.uchicago.edu/~klivescu)

**ACL 2019** &nbsp; **Best Paper Nominee**
[[paper]](https://aclanthology.org/P19-1180.pdf) 

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

Note: The demos are **NOT** used to reproduce the numbers reported in the ACL paper. 
To reproduce the results, we need to train multiple models and do self F1 driven model selection (see Section 4.7 for details).
To get roughly similar numbers to the ones reported in the paper, take the 6th or 7th checkpoint of VG-NSL and the checkpoints after 20 epochs for VG-NSL+HI. 
Thanks to [Nori](https://kojimano.github.io/) for bringing this into our attention. 

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
