## Amazon Rekognition Custom Labels Feedback

The Model Feedback solution enables you to give feedback on your model's predictions and make improvements by using human verification. Depending on the use case, you can be successful with a training dataset that has only a few images. A larger annotated training set might be required to enable you to build a more accurate model. The Model Feedback solution allows you to create larger dataset through model assistance.

The workflow for continuous model improvement is as follows:

- Train the first version of your model with a small training dataset.
- Provide an unannotated dataset to the Model Feedback solution.
- The Model Feedback solution uses the current model. It starts human verification jobs to annotate new dataset.
- Based on human feedback, the Model Feedback solution generates a manifest file that you use to create a new model.

## Deployment

Deploy CloudFormation stack in one of the AWS regions where you are using Amazon Rekognition Custom Labels.

Region| Launch
------|-----

US East (N. Virginia) | [![Deploy Feedback](http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/images/cloudformation-launch-stack-button.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?stackName=rekognition-custom-labels-feedback&templateURL=https://aws-workshops-us-east-1.s3.amazonaws.com/rekognition-feedback/cf-rekognition-feedback-use1.yaml)

EU West (Ireland) | [![Deploy Feedback](http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/images/cloudformation-launch-stack-button.png)](https://console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?stackName=rekognition-custom-labels-feedback&templateURL=https://aws-workshops-eu-west-1.s3.amazonaws.com/rekognition-feedback/cf-rekognition-feedback-use1.yaml)

## Configuration

After CloudFormation stack is deployed, click on Output tab and and make note of following. You will need these in the step below to run feedback client.

1. jobRoleArn
2. preLambdaArn
3. postLambdaArn

## Running Feedback Client

### PreReqs:
You can run feedback client from terminal with following installed:

- Python3 (https://www.python.org/downloads/)
- Pillow (https://pypi.org/project/Pillow/2.2.2/)
- AWS CLI (https://aws.amazon.com/cli/)

1. Go to Terminal (on your local desktop or EC2 instance etc.)
2. Type git clone https://github.com/aws-samples/amazon-rekognition-custom-labels-feedback-solution
4. Type cd amazon-rekognition-custom-labels-feedback-solution/src
5. Update feedback-config.json in src folder with values for your environment.
6. Run: python3 start-feedback.py

This will analyze images using projectVersionArn and start GroundTruth label verification jobs. You should see an ouput command that you can later use to generate manifest file for dataset.

7. After label verification jobs are complete in GroundTruth run the command you got in step 6.

This will generate dataset manifest file that you can use to train next version of your model in Amazon Rekognition Custom Labels.

## Cost

As you deploy this CloudFormation stack, it creates different resources (IAM roles, and AWS Lambda functions). You will get charged for different AWS resources created as part of the stack deployment. To avoid any recurring charges, delete stack.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

