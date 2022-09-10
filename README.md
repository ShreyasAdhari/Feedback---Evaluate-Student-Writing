# [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/c/feedback-prize-2021)

### Top 3% solution (53/2060) based on an ensemble of Transformer models. 

## Objective

- Objective is to automatically segment texts and classify argumentative and rhetorical elements in essays into 7 classes. 
- The 7 Classes are `Lead`, `Claim`, `CounterClaim`, `Rebuttal`, `Position`, `Evidence` and `Concluding Statement`.
- Metric used for evaluation was `F1 Score`.


## Requirements

- Python 3.9.7
- [Pytorch](https://pytorch.org/) 1.10.1
- [Transformers](https://huggingface.co/docs/transformers/index) 4.15.0


## Architecture

- The solution consists of an ensemble of 4 transformer models each containing 5 folds. Transformer model used were `Deberta V1`, `Deberta V2-Xlarge`, 
`Longformer` and `Funnel`. 
- `Token Clasification` approach was used and `AutoModelForSequenceClassification` class objects were used.
- Each model was trained using `Half-Precision` and `max_length = 1024`. 
- `Dynamic Padding` was used to reduce inference time drastically.
- Predictions from different models were averaged to give final result.

## Testing
- The above notebook uses a single model for prediction. The predictions for a single essay are then visualized.

## Inference

- [This](https://www.kaggle.com/code/shreyasadhari123/fast-inference-final) kaggle kernel was used for final submission.
