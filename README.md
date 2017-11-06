# VGG 11
|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | structure | acc | loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| SGD | O | O | O | GAP | X |  VGG 16  | 80.33% | 0.46 |
| SGD | O | O | O | FC | X |  VGG 16  | 82.00% | 0.44 |
| Momentum+ | O | O | O | FC | X |  VGG 11  | 82.33% | 0.417 | 
| Momentum+ | O | O | O | GAP | X |  VGG 11  | 82.21% | 0.43 |
| ADAM | O | O | O | GAP | X |  VGG 11  | ? | ? | 
| ADAM | O | O | O | GAP | X |  VGG 11  | ? | ? | 

# VGG 13
|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | structure | acc | loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| SGD | O | O | O | FC | X |  VGG 11  |  82.53%| 0.417 |
| SGD | O | O | O | GAP | X |  VGG 11  | 83.39%  | 0.412 |
| Momentum+ | O | O | O | FC | X |  VGG 13  | ? | ? | 
| Momentum+ | O | O | O | GAP | X |  VGG 13  | ? | ? |
| ADAM | O | O | O | GAP | X |  VGG 13  | ? | ? | 
| ADAM | O | O | O | GAP | X |  VGG 13  | ? | ? | 


# VGG 16
|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | structure | acc | loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SGD | O | O | O | FC | X |  VGG 13  | 83.8 | 0.411 | 
| SGD | O | O | O | GAP | X |  VGG 13  | 82.5 | 0.413 |
| Momentum+ | O | O | O | FC | X |  VGG 16  | ? | ? | 
| Momentum+ | O | O | O | GAP | X |  VGG 16  | ? | ? |
| ADAM | O | O | O | GAP | X |  VGG 16  | ? | ? | 
| ADAM | O | O | O | GAP | X |  VGG 16  | ? | ? | 



# VGG 19
|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | structure | acc | loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| SGD | O | O | O | FC | X |  VGG 19  | ? | ? | 
| SGD | O | O | O | GAP | X |  VGG 19  | ? | ? | 
| Momentum+ | O | O | O | FC | X |  VGG 19  | ? | ? |
| Momentum+ | O | O | O | GAP | X |  VGG 19  | ? | ? |
| ADAM | O | O | O | GAP | X |  VGG 19  | ? | ? | 
| ADAM | O | O | O | GAP | X |  VGG 19  | ? | ? |











