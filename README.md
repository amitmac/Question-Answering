### Question Answering

This code implements Dynamic Coattention Networks.

You can run the code by command -

python train.py --data_dir "[path]/data/squad/"

#### The model is yet to be evaluated properly.

Here is the loss convergence on dummy data of 32 data points after 10 epochs.

Loss after 0 epochs, 11.6004600525  
Loss after 1 epochs, 9.96237182617  
Loss after 2 epochs, 9.91187667847  
Loss after 3 epochs, 9.73106002808  
Loss after 4 epochs, 9.27107334137  
Loss after 5 epochs, 8.15467834473  
Loss after 6 epochs, 6.04854488373  
Loss after 7 epochs, 6.52948093414  
Loss after 8 epochs, 4.51838302612  
Loss after 9 epochs, 3.2650642395  

### Things to implement -
* Masking to avoid training using paddings

This is a part of assignment 4 of Stanford course CS224n. Some part of code was already given as a start point.