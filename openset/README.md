The openset classification challenge uses synthetic object images rendered from CAD models as the training domain and real object images cropped from the COCO dataset as the validation domain.

## Downloading Data

By downloading these datasets you agree to the following terms:

### Terms of Use
- You will use the data only for non-commercial research and educational purposes.
- You will NOT distribute the images.
- The organizers make no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
- You accept full responsibility for your use of the data.

You can download the datasets with 
    
    wget http://csr.bu.edu/ftp/visda17/openset/train.tar
    tar xvf train.tar
    
    wget http://csr.bu.edu/ftp/visda17/openset/validation.tar
    tar xvf validation.tar  
    
Images are structured in folders as 

- `{split}/{category}/{object_id}.jpg`

with a  single `image_list.txt` file in the root or each dataset that lists all images and corresponding labels for train/val subset. For test data, only images are provided. 

A technical report detailing the data generation process will be released in the near future. 

<!---
## Baselines and Rules

We have several baseline models with data readers in the [`/model`](model) folder. Each model has a short README on how to run it.

- "Adversarial Discriminative Domain Adaptation" (ADDA) with LeNet and VGG16 in Tensorflow [`arxiv`](https://arxiv.org/abs/1702.05464)
- "Learning Transferable Features with Deep Adaptation Networks" (DAN) with Alexnet in Caffe [`arxiv`](https://arxiv.org/pdf/1502.02791)
- "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" with Alexnet in Caffe [`arxiv`](https://arxiv.org/abs/1607.01719)

Please refer to the [challenge rules](http://ai.bu.edu/visda-2017/) for specific guidelines your method must follow.
-->

## Evaluating your Model

To evaluate the performance of your adaptation model, you should:
- Train you model with training data (with labels) and adapt it on the validation data (without labels).
- Predict labels for images in the validation set.
- Calculate the mean accuracies for each category and the overall mean of these accuracies. You are encouraged to upload your results to the evaluation server to compare your performance with that of other participants. 

The category IDs are as follows:
> 0 – aeroplane  
> 1 – bicycle  
> 2 – bus  
> 3 – car  
> 4 – horse  
> 5 – knife  
> 6 – motorcycle  
> 7 – person  
> 8 – plant  
> 9 – skateboard  
> 10 – train  
> 11 – truck  
> 12 – other (background class)  
 

Submissions will be evaluated by calculating the classification accuracy of each category and then the mean accuracy across all categories. The leaderboard on CodaLab will display all of these scores, and the official ranking will be determined by the mean classification accuracy across all categories. 

### Evaluation Server and Leaderboards
 
We are using CodaLab to evaluate results and host the leaderboards for this challenge. You can find the image classification competition [here](#). Please see the "Evaluation" tab in the competition for more details on leaderboard organization. 

### Feedback and Help
If you find any bugs please [open an issue](https://github.com/VisionLearningGroup/visda-2018-public/issues).


