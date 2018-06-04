The detection challenge uses synthetic object images rendered from CAD models as the training domain and real object images from the COCO dataset as the validation domain.

## Downloading Data

By downloading these datasets you agree to the following terms:

### Terms of Use
- You will use the data only for non-commercial research and educational purposes.
- You will NOT distribute the images.
- The organizers make no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
- You accept full responsibility for your use of the data.

You can download source and target datasets with 
    
    wget http://csr.bu.edu/ftp/visda/2018/detection/train.tar
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    
You might also want to download files containing ground truth boxes:

    wget http://csr.bu.edu/ftp/visda/2018/detection/visda18-detection-train.txt
    wget http://csr.bu.edu/ftp/visda/2018/detection/visda18-detection-test.txt
    wget http://csr.bu.edu/ftp/visda/2018/detection/coco17-train.txt
    wget http://csr.bu.edu/ftp/visda/2018/detection/coco17-val.txt

Last file only lists file names. 

A technical report detailing the data generation process will be released in the near future. 

We suggest using train and val splits of COCO17 as a target validation domain. We provide filtered ground truth for source data in two formats: 

- COCO-like annotations that can be found a single dataset.json file in the root
- `datalist.txt` format with every line following schema `{image_full_name} {xmin1} {ymin1} {xmax1} {ymax1} {class_id1} {xmin2} {ymin2} {xmax2} {ymax2} {class_id2}... ` i.e. number of bounding boxes can be computed as `(line.split()-1)/5`

<!---
## Baselines and Rules

We have several baseline models with data readers in the [`/model`](model) folder. Each model has a short README on how to run it.

- "Adversarial Discriminative Domain Adaptation" (ADDA) with LeNet and VGG16 in Tensorflow [`arxiv`](https://arxiv.org/abs/1702.05464)
- "Learning Transferable Features with Deep Adaptation Networks" (DAN) with Alexnet in Caffe [`arxiv`](https://arxiv.org/pdf/1502.02791)
- "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" with Alexnet in Caffe [`arxiv`](https://arxiv.org/abs/1607.01719)

Please refer to the [challenge rules](http://ai.bu.edu/visda-2017/) for specific guidelines your method must follow.
-->

## Evaluating your Model

We use the standard `IOU@0.5` metric for evaluation. Example evaluation script is avaliable in [`eval.py`](eval.py).

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
 
Please use the `datalist.txt` format for evaluation, specifically, each submission file should follow schema with *same* order of images as in `coco17-val.txt`: `{xmin1} {ymin1} {xmax1} {ymax1} {class_id1} {confidence1}...`  

### Evaluation Server and Leaderboards
 
We are using CodaLab to evaluate results and host the leaderboards for this challenge. You can find the image detection competition [here](https://competitions.codalab.org/competitions/18892). There are two competition leaderboards: the main leaderboard shows results of adapted models and will be used to determine the final team ranks. The expanded leaderboard additionally shows the team's source-only models, i.e. those trained only on the source domain without any adaptation. These results are useful for estimating how much the method improves upon its source-only model, but will not be used to determine team ranks.


### Submitting to the Evaluation Server
 
Once the servers become available, you will be able to submit your results:
- Generate "source_results.txt" and "adaptation_results.txt".
- Place these files into a zip file named [team_name]_submission
- Submit to CodaLab evaluation server following the instructions below

To submit your zipped result file to the appropriate VisDA Classification challenge click on the “Participate” tab. Select the phase (validation or testing). Select “Submit / View Results, fill in the required fields and click “Submit”. A pop-up will prompt you to select the results zip file for upload. After the file is uploaded, the evaluation server will begin processing. This might take some time. To view the status of your submission please select “Refresh Status”. If the status of your submission is “Failed” please check your file is named correctly and has the right format. You may refer to the scoring output and error logs for more details.

After you submit your results to the evaluation server, you can control whether your results are publicly posted to the CodaLab leaderboard. To toggle the public visibility of your results please select either “post to leaderboard” or “remove from leaderboard.” 


### Feedback and Help
If you find any bugs please [open an issue](https://github.com/VisionLearningGroup/visda-2018-public/issues).
