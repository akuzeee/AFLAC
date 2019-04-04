# Experiments for *Adversarial Invariant Feature Learning under Accuracy Constraint for Domain Generalization*


## Requirements
- python 2.7
- numpy==1.12.1
- pandas==0.20.1
- sacred==0.7.4 
- tensorboardX==1.2 
- torch==0.3.1
 

## Implementation
- `$ run.sh`
  - The MNISTR dataset will be automatically downloaded under `~/.torch/datasets/` by running the script.
- Then you can make a score table in ./analyze_results.ipynb
  

## References
- Ghifary, M., Bastiaan Kleijn, W., Zhang, M., Balduzzi, D.: Domain generaliza- tion for object recognition with multi-task autoencoders. In: Proc. of the IEEE International Conference on Computer Vision (ICCV) (2015)
  - We borrow the dataset from their repository https://github.com/ghif/mtae
