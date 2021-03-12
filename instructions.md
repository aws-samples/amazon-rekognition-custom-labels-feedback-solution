Light Weighting Data Annotation and Model Training Tasks with Amazon Rekognition Custom Labels Feedback Solution

Background: 

For supervised machine learning problem, labels are values expected to be learned and predicted by a model. To obtain first principal labels, machine learning practitioners can either record them in operation or conduct data annotation, which are activities that assign labels to the dataset based on human intelligence. However, manual dataset annotation can be tedious and tiring for human, especially on a large dataset. Even with obvious labels to annotate it can still be error-prone. To tackle this issue, in this post we demonstrate how to use an assisting machine learning model to speed up the annotation while have human-in-the-loop (HITL). As an example, we focus on a Computer Vision object detection setting. We’ll detect AWS/Amazon smile logos from images collected on the AWS and Amazon website. Depending on the use-case, one can start with training a model with only a few images that captures the obvious pattern in the dataset, and have human focus on lightweight tasks of reviewing these automatically proposed annotation, and adjust mistaken labels only when necessary. As a result, it avoids repeating manual work, reduces human fatigue, and therefore improves data annotation quality and efficiency.

In this post we will use AWS CloudFormation to setup serverless stack with lambda functions as well as their corresponding permissions, Amazon S3 for image data lake and model prediction storage, Amazon SageMaker Ground Truth for data labeling, and Amazon Rekognition Custom Labels for dataset management and model training/hosting. Code used in this post is available on GitHub (https://github.com/aws-samples/amazon-rekognition-custom-labels-feedback-solution) repository.

Get Started:

We will demonstrate an end-to-end solution detecting Amazon smile logo in different images.

Preparing: 

*S3 bucket with images*
First, create a new S3 bucket in the designed region (N. Virginia or Ireland) with 2 partitions: one with smaller number of images, and another with larger number. For example, in this post, s3://rekognition-custom-labels-feedback-amazon-logo/v1/train/ includes 8 AWS/Amazon smile logo images;
[Image: s3-data-v1.png]and s3://rekognition-custom-labels-feedback-amazon-logo/v2/train/ has 20 AWS/Amazon smile logo images.
[Image: s3-data-v2.png]
*GT Labeling Workforce*
In this post we leverage Amazon SageMaker Ground Truth private labeling workforce. We will also add new “workers”, create new “private team”, and add the worker into the team. Eventually, we need to take record of the labelling workforce Amazon Resource Name (ARN).
[Image: image]To add new worker, click “invite new workers”, which enables you to add the email address of new workers.
[Image: 1.png]Once added, the worker will receive an email, by default similar to the one shown as below.
[Image: 2.png]Meanwhile, you can create a new labeling team by clicking “create private team”, and fill out the team name and confirm.
[Image: 3.png]Then click the name of the labeling team, select “Workers” tab, and “Add workers to team”.

[Image: 4.png]Select intended workers’ Email address and confirm “Add workers to team”.
[Image: 5.png]Finally we can get the labeling workforce ARN. For more details see Amazon SageMaker Ground Truth Create and Manage Workforces (https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-management.html).

*Preinstall packages on your terminal*
Please make sure Python3 (https://www.python.org/downloads/), Pillow (https://pypi.org/project/Pillow/2.2.2/), and AWS CLI (https://aws.amazon.com/cli/) are installed on your environment and AWS profile configuration and credentials are setup, before you move to next steps.

Detailed Steps of the Solution 

1. Deploy CloudFormation stack

Deploy CloudFormation stack in one of the AWS regions where you are going to use Amazon Rekognition Custom Labels. This solution is currently available in us-east-1 (N. Virginia) and eu-west-1 (Ireland) regions. 
[Image: cloudformation-launch-stack.png]

Region	Launch
US East (N. Virginia)	Launch stack (https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?stackName=rekognition-custom-labels-feedback&templateURL=https://aws-workshops-us-east-1.s3.amazonaws.com/rekognition-feedback/cf-rekognition-feedback-use1.yaml)
EU West (Ireland)	Launch stack (https://console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?stackName=rekognition-custom-labels-feedback&templateURL=https://aws-workshops-eu-west-1.s3.amazonaws.com/rekognition-feedback/cf-rekognition-feedback-euw1.yaml)

After deployment, click “Outputs” tab and make note of the three outputs: jobRoleArn, preLambdaArn, and postLambdaArn.
[Image: cloudformation-output.png]
1. Train first version of your model in Rekognition Custom Labels with the smaller dataset

Please refer here (https://aws.amazon.com/blogs/machine-learning/announcing-amazon-rekognition-custom-labels/) for how to create a project and train a model using Amazon Rekognition Custom Labels. In this post, we created a project called “custom-labels-feedback”. The first version model was trained and validated using v1 dataset that includes 8 AWS/Amazon smile logo images. Here are some labeled sample data used for training:
[Image: rek-dataset-v1.png]When the first version model’s training process is finished, take a note of your model ARN. In our example, the model performance achieved an F1 score as 0.667. We’ll use this model to help human workers to annotate a larger dataset (v2) for next iteration.
[Image: initial-model.png]
1. Start feedback client

Go to terminal and git clone repository:  

git clone https://github.com/aws-samples/amazon-rekognition-custom-labels-feedback-solution.git 

Then change working directory:

cd amazon-rekognition-custom-labels-feedback-solution/src

Update feedback-config.json in src/ folder. Items need to be updated are:

* “images” — S3 bucket folder that has the larger dataset. In this post, it is the v2 dataset that contains 30 Amazon logo images.
* “outputBucket” — Output S3 bucket. For best practices, it is recommended to use the same image bucket here.
* “jobRoleArn” — Output from CloudFormation as noted above.
* “workforceTeamArn” — Private team ARN as set above in SageMaker Ground Truth Labeling workforces.
* “preLambdaArn” — Output from CloudFormation as noted above.
* “postLambdaArn” — Output from CloudFormation as noted above.
* “projectVersionArn” — The first model ARN as noted above

[Image: json-file.png]Start the first version model before you call feedback client. To do this, expand API Code section on your Rekognition Custom Labels model page, and simply copy/paste the AWS CLI command of “Start model” to your terminal:
[Image: start-model.png]The model status is then changing to “STARTING”. After the model status changes to “RUNNING”, run code in your terminal:

python3 start-feedback.py

It analyzes the larger dataset of images using the first version model and starts Ground Truth label verification jobs. It also outputs a command for later usage, which will generate manifest file for the larger dataset (e.g., v2 dataset). 
[Image: start-feedback-output.png]Now human workers can log in the labeling project to verify labels proposed by the first version model. Usually, label verification jobs will be sent to the workers in several batches. 
[Image: label-job.png]For most of the images that are labeled correctly by the first version model, human workers only need to confirm these labeling without any adjustment, which largely accelerates the whole data annotation process.
[Image: label-confirm.png]After label verification jobs are complete, i.e., status of Labeling jobs in Amazon SageMaker Ground Truth changes from “In progress” to “Complete”, run the command:

python3 get-feedback.py --jobs-manifest s3://......

that you got from above feedback client’s output in your terminal. It will generate manifest file for the larger dataset that you can use to train next version of your model in Amazon Rekognition Custom Labels. The output “S3 Path” indicates manifest file location for the larger dataset.
[Image: get-feedback-output.png]
1. Train next version of your model in Rekognition Custom Labels

Create a new dataset in Amazon Rekognition Custom Labels. In “Image location”, choose “Import images labeled by Amazon SageMaker Ground Truth”, and put the above noted output “S3 Path” in “.manifest file location”. 
[Image: add-dataset-manifest.png]Double check if all images are labeled correctly. Here are some sample data that we imported from SageMaker Ground Truth:
[Image: rek-dataset-v2.png]Using this newly added dataset in Rekognition Custom Labels, you can train next version of your model under the project same as the first version’s. For example, in this post, we train next version model using dataset “amazon-logo-v2” under project “custom-labels-feedback”, and use dataset “amazon-logo-v1” as a test set.
[Image: train-next-version.png]In our example, comparing to the first version, the next version model achieves much better performance with a 0.900 F1 score. It is worth noting that this solution can be applied multiple times in a Rekognition Custom Labels project. You can use the next version model to easily annotate even larger datasets and train models until you are satisfied with final model performance.
[Image: next-verion-model.png]
Clean up 

After you finish using the custom labels feedback solution, remember to delete the CloudFormation stack in your AWS console, and stop running models by calling AWS CLI command in your terminal to avoid unnecessary charges.

Conclusion 

This post presented an end-to-end demonstration of leveraging Amazon Rekognition Custom Labels to efficiently annotate larger dataset with assistance from a model trained on smaller dataset. This solution enables users to gain feedback on model’s performance and make improvements by using human verification and adjustment when necessary. As a result, data annotation, model training and error analysis are conducted simultaneously and interactively which improves dataset annotation efficiency.

