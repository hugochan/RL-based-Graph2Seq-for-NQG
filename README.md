# RL-based-Graph2Seq-for-NQG
Code & data accompanying the ICLR 2020 paper "Reinforcement Learning Based Graph-to-Sequence Model for Natural Question Generation"

## Get started


### Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.



### Run the model

* Download the preprocessed data from [squad-split1](https://drive.google.com/drive/folders/1EoIqyqaSIsES_MrsKnHNx4SYRIxom0YD) and [squad-split2](https://drive.google.com/drive/folders/11gett0qzTW2SvNjjcRLJik-FommhP8bs). And put the data under the root directory. So the file hierarchy will be like: `data/squad-split1` and `data/squad-split2`. 


<!--(Note: if you cannot access the above data, please download from [here](http://academic.hugochan.net/download/graphflow-data.zip).)
-->

* Run the model

    ```
    python main.py -config config/squad_split1/graph2seq_static_bert_finetune_word_70k_0.4_bs_60.yml
    ```
  	Note that you can specify the output path by modifying `out_dir` in a config file. 
  	If you want to finetune a pretrained model, you can specify the path to the pretrained model by modifying `pretrained` and you need to set `out_dir ` to null.
  	If you just want to load a pretrained model and evaluate it on a test set, you need to set both `trainset` and `devset` to null.
  	
    

* Finetune the model using RL

    ```
    python main.py -config config/squad_split1/rl_graph2seq_static_bert_finetune_word_70k_0.4_bs_60.yml
    ```



## Reference

If you found this code useful, please consider citing the following paper:

Yu Chen, Lingfei Wu and Mohammed J. Zaki. **"Reinforcement Learning Based Graph-to-Sequence Model for Natural Question Generation."** In *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020), Addis Ababa, Ethiopia, Apr 26-30, 2020.*


	@article{chen2019reinforcement,
	  title={Reinforcement learning based graph-to-sequence model for natural question generation},
	  author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed J},
	  journal={arXiv preprint arXiv:1908.04942},
	  year={2019}
	}
	
