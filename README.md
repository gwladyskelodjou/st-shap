## ST-SHAP : A more stable approach of Kernel SHAP

This repository contains the code of ST-SHAP, an approach to improving the stability of Kernel SHAP by modifying the strategy for generating neighbours.

## Idea behind ST-SHAP
The idea to enhance stability is to fill the layers in increasing order as long as there is still enough budget. Unlike SHAP, if the budget is not sufficient to fill a layer, random generation only takes place in that layer.

## Use
ST-SHAP is utilized in a manner similar to Kernel SHAP, with the inclusion of an additional binary parameter named ``review``. When set to ``True``, it signifies the application of the ST-SHAP neighbor selection process; otherwise, it corresponds to the Kernel SHAP process.

## Notebook
You can find an example of how to use ST-SHAP in notebook.ipynb